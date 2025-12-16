"""
This script, `tag-genie`, is a command-line interface (CLI) tool for
automating the categorization and auditing of business directory listings.

It provides three main functionalities:
1.  PROCESS: Classify raw text data using a zero-shot AI model.
2.  AUDIT: Analyze the results of a processing run to determine accuracy and risk.
3.  CLEAN: Apply a compliance policy to the audited data to generate a final,
    clean dataset for production use.
"""
import typer
import csv
import sys
from rich.console import Console
from rich.table import Table
from rich.progress import Progress
from transformers import pipeline
from functools import lru_cache
from typing import List, Dict, Any

# --- Main Typer App and Console ---
app = typer.Typer()
console = Console()

# --- Shared Constants and Mappings ---
NONE_TAG = "None of the above"

# MAPPING: Long AI Tags -> Original Short Categories (for comparison in audit/clean)
TAG_MAP = {
    "Legal Services and Immigration Consultants": "Legal & Immigration",
    "Chartered Accountants and Tax Consultants": "Finance & Tax",
    "Relocation Services and Lifestyle Management": "Relocation & Lifestyle",
    "Real Estate Agency and Property Rentals": "Real Estate",
    "None of the above": "None"
}

# MAPPING: Long AI Tags -> Final Short DB-friendly format (for final output in clean)
SHORT_TAG_MAP = {
    "Legal Services and Immigration Consultants": "Legal",
    "Chartered Accountants and Tax Consultants": "Finance",
    "Relocation Services and Lifestyle Management": "Lifestyle",
    "Real Estate Agency and Property Rentals": "Real Estate",
    "None of the above": "None"
}

# ==============================================================================
# 1. PROCESS COMMAND (Original `process-csv` functionality)
# ==============================================================================

@lru_cache(maxsize=None)
def get_classifier():
    """Loads and caches the zero-shot classification pipeline."""
    with console.status("[bold green]Loading classification model (this may take a moment)...", spinner="dots"):
        return pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

def run_classification(text: str, candidate_tags: List[str]) -> Dict[str, Any]:
    """Runs the zero-shot classification and returns processed results."""
    text = text.strip()
    if not text:
        return {"winner_tag": NONE_TAG, "winner_score": 1.0}

    classifier = get_classifier()
    full_tag_list = candidate_tags + [NONE_TAG]
    result = classifier(text, full_tag_list, multi_label=False)
    
    return {
        "winner_tag": result["labels"][0],
        "winner_score": result["scores"][0]
    }

@app.command(name="process", help="Classify raw text from a CSV file using an AI model.")
def process_csv(
    input_file: str = typer.Argument(..., help="Path to the source CSV file."),
    output_file: str = typer.Argument(..., help="Path to save the processed CSV file."),
    column: str = typer.Argument(..., help="The name of the column containing the text to classify."),
    tags: str = typer.Option(..., "--tags", help="A comma-separated list of candidate tags."),
):
    """Processes a CSV file to classify text in a specified COLUMN using TAGS."""
    console.print(f"[bold]Starting batch processing for [cyan]{input_file}[/cyan]...[/bold]")
    candidate_tags = [tag.strip() for tag in tags.split(',')]
    
    try:
        with open(input_file, 'r', encoding='utf-8-sig') as infile:
            reader = csv.DictReader(infile)
            original_headers = reader.fieldnames
            if column not in original_headers:
                console.print(f"[bold red]Error:[/bold red] Column '{column}' not found in the CSV header.")
                raise typer.Exit()
            
            new_headers = original_headers + ["Predicted_Tag", "Confidence_Score"]
            
            # Read all rows into memory to count for progress bar
            rows = list(reader)
            total_rows = len(rows)

            with open(output_file, 'w', newline='', encoding='utf-8') as outfile:
                writer = csv.DictWriter(outfile, fieldnames=new_headers)
                writer.writeheader()

                with Progress() as progress:
                    task = progress.add_task("[green]Processing rows...", total=total_rows)
                    for row in rows:
                        text_to_classify = row[column]
                        try:
                            result_data = run_classification(text_to_classify, candidate_tags)
                            row["Predicted_Tag"] = result_data['winner_tag']
                            row["Confidence_Score"] = result_data['winner_score']
                        except Exception as e:
                            console.print(f"[bold red]Error processing row:[/bold red] {e}")
                            row["Predicted_Tag"] = "ERROR"
                            row["Confidence_Score"] = "0.0"
                        finally:
                            writer.writerow(row)
                            progress.update(task, advance=1)
            
            console.print(f"\n[bold green]Success![/bold green] Processed file saved to [cyan]{output_file}[/cyan].")

    except FileNotFoundError:
        console.print(f"[bold red]Error:[/bold red] Input file '{input_file}' not found.")
        raise typer.Exit()

# ==============================================================================
# 2. AUDIT COMMAND (from `analyze_risk_v2.py`)
# ==============================================================================

@app.command(name="audit", help="Analyze a processed CSV to calculate agreement rates and find high-risk mismatches.")
def audit(
    input_path: str = typer.Argument(..., help="Path to the processed CSV file (e.g., the output of the 'process' command).")
):
    """Analyzes a classification audit file to calculate agreement rates and identify risks."""
    total, agreements, high_conf_agreements, high_conf_disagreements, low_conf = 0, 0, 0, 0, 0
    danger_rows = []

    try:
        with open(input_path, 'r', encoding='utf-8-sig') as f:
            reader = csv.DictReader(f)
            for row in reader:
                total += 1
                original = row.get('Category', '').strip()
                predicted_long = row.get('Predicted_Tag', '').strip()
                try:
                    confidence = float(row.get('Confidence_Score', 0.0))
                except (ValueError, TypeError):
                    confidence = 0.0
                
                predicted_short = TAG_MAP.get(predicted_long, predicted_long)
                is_match = (original == predicted_short)
                
                if is_match:
                    agreements += 1
                    if confidence > 0.8:
                        high_conf_agreements += 1
                else:
                    if confidence > 0.8:
                        high_conf_disagreements += 1
                        danger_rows.append({
                            "Name": row.get('Name'),
                            "Original": original,
                            "Predicted": predicted_long,
                            "Conf": confidence
                        })
                if confidence < 0.5:
                    low_conf += 1
        
        console.print(f"\n[bold blue]🛡️  Compliance Audit Report: {input_path}[/bold blue]\n")
        table = Table(title="Risk Analysis Summary")
        table.add_column("Metric", style="cyan")
        table.add_column("Count", justify="right")
        table.add_column("Percentage", justify="right", style="green")

        table.add_row("Total Rows", str(total), "100%")
        table.add_row("Total Agreement", str(agreements), f"{(agreements/total)*100:.1f}%")
        table.add_row("High Confidence Agreement", str(high_conf_agreements), f"{(high_conf_agreements/total)*100:.1f}%")
        table.add_row("Low Confidence (< 50%)", str(low_conf), f"{(low_conf/total)*100:.1f}%")
        table.add_row("[red]Danger Zone (High Conf Mismatch)[/red]", str(high_conf_disagreements), f"[red]{(high_conf_disagreements/total)*100:.1f}%[/red]")
        console.print(table)

        if danger_rows:
            console.print("\n[bold red]🚨 DANGER ZONE EXAMPLES (High Confidence Mismatches)[/bold red]")
            danger_table = Table(show_header=True, header_style="bold red")
            danger_table.add_column("Name")
            danger_table.add_column("Original")
            danger_table.add_column("AI Prediction")
            danger_table.add_column("Conf")
            for r in danger_rows[:10]:
                danger_table.add_row(r['Name'][:30], r['Original'], r['Predicted'].replace(" and ", "\n& "), f"{r['Conf']:.2f}")
            console.print(danger_table)
            if len(danger_rows) > 10:
                console.print(f"...and {len(danger_rows) - 10} more rows.")

    except FileNotFoundError:
        console.print(f"[bold red]Error:[/bold red] Input file '{input_path}' not found.")
        raise typer.Exit()


# ==============================================================================
# 3. CLEAN COMMAND (from `apply_fixes_v2.py`)
# ==============================================================================

@app.command(name="clean", help="Apply a compliance policy to an audited CSV to generate a final, clean dataset.")
def clean(
    input_path: str = typer.Argument(..., help="Path to the audited CSV file."),
    output_path: str = typer.Argument(..., help="Path to save the final clean CSV file.")
):
    """Applies a compliance policy to an audited CSV to generate a final, clean dataset."""
    stats = {"auto_fixed": 0, "verified": 0, "low_conf": 0, "total": 0}
    
    try:
        with open(input_path, 'r', encoding='utf-8-sig') as fin, \
             open(output_path, 'w', encoding='utf-8', newline='') as fout:
            
            reader = csv.DictReader(fin)
            fieldnames = reader.fieldnames + ['Audit_Status', 'Final_Tag']
            writer = csv.DictWriter(fout, fieldnames=fieldnames)
            writer.writeheader()

            for row in reader:
                stats["total"] += 1
                original = row.get('Category', '').strip()
                predicted_long = row.get('Predicted_Tag', '').strip()
                try:
                    confidence = float(row.get('Confidence_Score', 0.0))
                except (ValueError, TypeError):
                    confidence = 0.0
                
                predicted_short_for_comparison = TAG_MAP.get(predicted_long, predicted_long)
                
                if confidence > 0.85 and original != predicted_short_for_comparison:
                    final_tag = SHORT_TAG_MAP.get(predicted_long)
                    if final_tag and final_tag != "None":
                        row['Final_Tag'] = final_tag
                        row['Audit_Status'] = "AUTO_FIXED"
                        stats["auto_fixed"] += 1
                    else:
                        row['Final_Tag'] = original
                        row['Audit_Status'] = "VERIFIED"
                        stats["verified"] += 1
                elif confidence < 0.5:
                    row['Final_Tag'] = original
                    row['Audit_Status'] = "NEEDS_REVIEW"
                    stats["low_conf"] += 1
                else:
                    row['Final_Tag'] = original
                    row['Audit_Status'] = "VERIFIED"
                    stats["verified"] += 1
                writer.writerow(row)

        console.print(f"\n[bold green]✅ Audit Complete. Fixes applied to: {output_path}[/bold green]")
        summary_table = Table(title="File Processing Summary")
        summary_table.add_column("Metric", style="cyan")
        summary_table.add_column("Count", justify="right")
        summary_table.add_column("Percentage", justify="right", style="green")

        summary_table.add_row("Total Rows", str(stats['total']), "100%")
        summary_table.add_row("Verified (Kept Original)", str(stats['verified']), f"{(stats['verified']/stats['total'])*100:.1f}%")
        summary_table.add_row("Low Confidence (Needs Review)", str(stats['low_conf']), f"{(stats['low_conf']/stats['total'])*100:.1f}%")
        summary_table.add_row("Auto-Fixed (High Risk Errors)", f"[bold red]{stats['auto_fixed']}[/bold red]", f"[bold red]{(stats['auto_fixed']/stats['total'])*100:.1f}%")
        console.print(summary_table)

    except FileNotFoundError:
        console.print(f"[bold red]Error:[/bold red] Input file '{input_path}' not found.")
        raise typer.Exit()


if __name__ == "__main__":
    app()
"""
Create Filtering Statistics Tables
-----------------------------------

This script reads the filtering_statistics.txt file and creates tables
showing branch counts for each filtering stage.
Each table is saved as a PNG image.
"""

import os
import re
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.table import Table

def parse_statistics_file(file_path):
    """
    Parse the filtering_statistics.txt file and extract branch counts.
    
    Returns:
        dict: Dictionary containing branch counts for each stage and model
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    stats = {}
    models = ['TypeA', 'TypeB', 'TypeC', 'TypeD', 'TypeE', 'modified_model']
    
    # Initialize stats dictionary
    for model in models:
        stats[model] = {}
    
    # Parse main section (Filter #6-8)
    for model in models:
        # Pattern to match model section and extract all numbers (more flexible with whitespace)
        pattern = rf"{model}:\s*\n(?:[^\n]*\n)*?\s*Before filter #6 \(initial branch count\):\s*(\d+)"
        match = re.search(pattern, content)
        if match:
            stats[model]['before_filter6'] = int(match.group(1))
        
        pattern = rf"{model}:\s*\n(?:[^\n]*\n)*?\s*After filter #6 \(after distance filter\):\s*(\d+)"
        match = re.search(pattern, content)
        if match:
            stats[model]['after_filter6'] = int(match.group(1))
        
        pattern = rf"{model}:\s*\n(?:[^\n]*\n)*?\s*After filter #7 \(after removing overlaps\):\s*(\d+)"
        match = re.search(pattern, content)
        if match:
            stats[model]['after_filter7'] = int(match.group(1))
        
        pattern = rf"{model}:\s*\n(?:[^\n]*\n)*?\s*After filter #8 \(after selecting top N\):\s*(\d+)"
        match = re.search(pattern, content)
        if match:
            stats[model]['after_filter8'] = int(match.group(1))
    
    # Parse graph filter section
    graph_section = re.search(r'Additional Filtering in 5graph_distance_map\.py.*?(?=Summary|$)', content, re.DOTALL)
    if graph_section:
        graph_content = graph_section.group(0)
        for model in models:
            pattern = rf"{model}:\s*\n\s*Before filter \(node_0 condition\):\s*(\d+)"
            match = re.search(pattern, graph_content)
            if match:
                stats[model]['before_graph_filter'] = int(match.group(1))
            
            pattern = rf"{model}:\s*\n(?:[^\n]*\n)*?\s*After filter \(node_0 in \[10, 20\)\):\s*(\d+)"
            match = re.search(pattern, graph_content)
            if match:
                stats[model]['after_graph_filter'] = int(match.group(1))
    
    return stats


def create_table_image(data, title, output_path):
    """
    Create a table image from data and save as PNG.
    
    Args:
        data: Dictionary with model names as keys and counts as values
        title: Title for the table
        output_path: Path to save the PNG file
    """
    models = ['TypeA', 'TypeB', 'TypeC', 'TypeD', 'TypeE', 'modified_model']
    
    # Prepare table data: 2 rows
    # Row 1: Model names
    # Row 2: Branch counts
    table_data = []
    
    # First row: model names
    model_row = []
    for model in models:
        model_row.append(model)
    table_data.append(model_row)
    
    # Second row: branch counts
    count_row = []
    for model in models:
        count = data.get(model, 'N/A')
        if isinstance(count, (int, float)):
            count_row.append(f"{count:,}")
        else:
            count_row.append(str(count))
    table_data.append(count_row)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 2))
    ax.axis('off')
    
    # Create table - simple 2 rows, no column headers
    table = ax.table(
        cellText=table_data,
        rowLabels=['Model', 'Branch Count'],
        cellLoc='center',
        loc='center',
        bbox=[0, 0, 1, 1]
    )
    
    # Style the table
    table.auto_set_font_size(False)
    table.set_fontsize(13)
    table.scale(1, 2.5)
    
    # Style row labels (left column)
    for i in range(2):
        cell = table[(i, -1)]
        cell.set_facecolor('#4CAF50')
        cell.set_text_props(weight='bold', color='white')
        cell.set_width(0.2)
    
    # Style first row (model names)
    for j in range(len(models)):
        cell = table[(0, j)]
        cell.set_facecolor('#E8F5E9')
        cell.set_text_props(weight='bold')
    
    # Style second row (branch counts)
    for j in range(len(models)):
        cell = table[(1, j)]
        cell.set_facecolor('#F5F5F5')
    
    # Add title
    fig.suptitle(title, fontsize=16, fontweight='bold', y=0.92)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"Saved: {output_path}")


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    stats_file = os.path.join(script_dir, "3distance_map_csv", "filtering_statistics.txt")
    output_dir = os.path.join(script_dir, "filtering_tables")
    os.makedirs(output_dir, exist_ok=True)
    
    if not os.path.exists(stats_file):
        print(f"Error: {stats_file} not found!")
        return
    
    # Parse statistics
    stats = parse_statistics_file(stats_file)
    
    # Debug: print parsed statistics
    print("Parsed statistics:")
    for model in ['TypeA', 'TypeB', 'TypeC', 'TypeD', 'TypeE', 'modified_model']:
        print(f"  {model}: {stats[model]}")
    print()
    
    # Define table stages
    stages = [
        {
            'key': 'before_filter6',
            'title': 'Before Filter #6\n(Initial Branch Count)',
            'filename': 'table_before_filter6.png'
        },
        {
            'key': 'after_filter6',
            'title': 'After Filter #6\n(After Distance Filter)',
            'filename': 'table_after_filter6.png'
        },
        {
            'key': 'after_filter7',
            'title': 'After Filter #7\n(After Removing Overlaps)',
            'filename': 'table_after_filter7.png'
        },
        {
            'key': 'after_filter8',
            'title': 'After Filter #8\n(After Selecting Top N)',
            'filename': 'table_after_filter8.png'
        },
        {
            'key': 'before_graph_filter',
            'title': 'Before Graph Filter\n(node_0 Condition)',
            'filename': 'table_before_graph_filter.png'
        },
        {
            'key': 'after_graph_filter',
            'title': 'After Graph Filter\n(node_0 in [10, 20))',
            'filename': 'table_after_graph_filter.png'
        }
    ]
    
    # Create tables for each stage
    for stage in stages:
        key = stage['key']
        title = stage['title']
        filename = stage['filename']
        
        # Extract data for this stage
        stage_data = {}
        for model in ['TypeA', 'TypeB', 'TypeC', 'TypeD', 'TypeE', 'modified_model']:
            stage_data[model] = stats[model].get(key, 'N/A')
        
        # Only create table if at least one model has data
        has_data = any(isinstance(v, int) for v in stage_data.values())
        if has_data:
            output_path = os.path.join(output_dir, filename)
            create_table_image(stage_data, title, output_path)
        else:
            print(f"Skipping {filename}: No data available (data: {stage_data})")
    
    print(f"\nAll tables saved to: {output_dir}")


if __name__ == "__main__":
    main()


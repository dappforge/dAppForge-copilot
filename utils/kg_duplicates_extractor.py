import json
import os
import csv
from collections import Counter
from typing import Dict, List, Tuple
import re
import pandas as pd

def extract_relationship_types(file_path: str) -> Dict[str, int]:
    """
    Extract all unique relationship types (edge types) from a graph store JSON file
    Returns a dictionary mapping relationship type to count
    """
    edge_counts = Counter()
    
    print(f"Reading file: {file_path}")
    
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
            
            if not isinstance(data, dict) or 'graph_dict' not in data:
                print(f"Error: Invalid graph store format in {file_path}")
                return {}
            
            graph = data['graph_dict']
            print(f"Processing {len(graph)} base nodes...")
            
            # Process each node and its edges
            for source_node, relations in graph.items():
                if isinstance(relations, list):
                    for relation in relations:
                        if isinstance(relation, list) and len(relation) == 2:
                            edge_type, target_node = relation
                            edge_counts[edge_type] += 1
    
    except Exception as e:
        print(f"Error processing {file_path}: {str(e)}")
        return {}
    
    print(f"Found {len(edge_counts)} unique relationship types")
    return dict(edge_counts)

def normalize_label(label: str) -> str:
    """
    Normalize a label for similarity comparison
    """
    # Convert to lowercase and remove extra whitespace
    normalized = label.lower().strip()
    # Remove common punctuation
    normalized = re.sub(r'[^\w\s]', '', normalized)
    # Replace multiple spaces with single space
    normalized = re.sub(r'\s+', ' ', normalized)
    return normalized

def find_similar_labels(edge_counts: Dict[str, int], similarity_threshold: float = 0.8) -> List[List[str]]:
    """
    Find groups of similar labels based on normalized text similarity
    Returns list of groups, where each group contains similar labels
    """
    labels = list(edge_counts.keys())
    groups = []
    processed = set()
    
    for i, label1 in enumerate(labels):
        if label1 in processed:
            continue
            
        group = [label1]
        processed.add(label1)
        
        norm1 = normalize_label(label1)
        
        for j, label2 in enumerate(labels[i+1:], i+1):
            if label2 in processed:
                continue
                
            norm2 = normalize_label(label2)
            
            # More precise similarity check
            # Only group if they are very similar (not just containing the same word)
            if (norm1 == norm2 or 
                (len(norm1) > 3 and len(norm2) > 3 and 
                 (norm1.startswith(norm2) or norm2.startswith(norm1)) and
                 abs(len(norm1) - len(norm2)) <= 5)):  # Length difference should be small
                group.append(label2)
                processed.add(label2)
        
        if len(group) > 1:  # Only include groups with duplicates
            groups.append(group)
    
    return groups

def suggest_canonical_name(group: List[str]) -> str:
    """
    Suggest a canonical name for a group of similar labels
    """
    if not group:
        return ""
    
    # Find the shortest label as base
    shortest = min(group, key=len)
    
    # Common patterns for canonical names
    patterns = {
        'allows': 'ALLOWS',
        'has': 'HAS', 
        'is': 'IS',
        'provides': 'PROVIDES',
        'uses': 'USES',
        'contains': 'CONTAINS',
        'defines': 'DEFINES',
        'implements': 'IMPLEMENTS',
        'calls': 'CALLS',
        'returns': 'RETURNS',
        'version': 'VERSION',
        'dependency': 'DEPENDENCY'
    }
    
    # Check if any pattern matches
    shortest_lower = shortest.lower()
    for pattern, canonical in patterns.items():
        if pattern in shortest_lower:
            return canonical
    
    # If no pattern matches, use the shortest label in uppercase
    return shortest.upper()

def analyze_similar_groups(similar_groups: List[List[str]], edge_counts: Dict[str, int]) -> List[Dict]:
    """
    Analyze similar groups and provide detailed information for manual review
    """
    analysis = []
    
    for i, group in enumerate(similar_groups, 1):
        group_info = {
            'Group_ID': f'Group_{i}',
            'Labels': ' | '.join(group),
            'Total_Count': sum(edge_counts[label] for label in group),
            'Proposed_Canonical': suggest_canonical_name(group),
            'Analysis': '',
            'Recommendation': ''
        }
        
        # Analyze the group
        shortest = min(group, key=len)
        longest = max(group, key=len)
        
        if len(group) == 2 and shortest.lower() in longest.lower():
            group_info['Analysis'] = f'"{shortest}" appears to be a shorter version of "{longest}"'
            group_info['Recommendation'] = f'Consider merging to "{shortest.upper()}" if they mean the same thing'
        elif all('allows' in label.lower() for label in group):
            group_info['Analysis'] = 'All labels contain "allows" - likely permission/capability relationships'
            group_info['Recommendation'] = 'Review if these represent different types of permissions or can be unified'
        elif all('has' in label.lower() for label in group):
            group_info['Analysis'] = 'All labels contain "has" - likely possession/attribute relationships'
            group_info['Recommendation'] = 'Review if these represent different types of attributes or can be unified'
        elif all('is' in label.lower() for label in group):
            group_info['Analysis'] = 'All labels contain "is" - likely identity/classification relationships'
            group_info['Recommendation'] = 'Review if these represent different types of classifications or can be unified'
        else:
            group_info['Analysis'] = 'Labels share common words but may have different semantic meanings'
            group_info['Recommendation'] = 'Manual review required - check if these truly represent the same relationship type'
        
        analysis.append(group_info)
    
    return analysis

def generate_excel_file(kg_type: str, edge_counts: Dict[str, int], output_dir: str):
    """
    Generate Excel file with multiple sheets for duplicates analysis
    """
    # Sort by count descending
    sorted_edges = sorted(edge_counts.items(), key=lambda x: x[1], reverse=True)
    
    # Create Excel file path
    excel_path = os.path.join(output_dir, f"{kg_type}_duplicates_analysis.xlsx")
    
    with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
        # Sheet 1: Main duplicates sheet
        duplicates_data = []
        for label, count in sorted_edges:
            duplicates_data.append([label, count, '', '', ''])
        
        duplicates_df = pd.DataFrame(duplicates_data, columns=['Label', 'Count', 'Similar_Labels', 'Proposed_Canonical', 'Notes'])
        duplicates_df.to_excel(writer, sheet_name='Duplicates', index=False)
        
        # Sheet 2: Similar groups
        similar_groups = find_similar_labels(edge_counts)
        if similar_groups:
            groups_data = []
            for i, group in enumerate(similar_groups, 1):
                total_count = sum(edge_counts[label] for label in group)
                labels_str = ' | '.join(group)
                suggested_canonical = suggest_canonical_name(group)
                groups_data.append([f'Group_{i}', labels_str, total_count, suggested_canonical])
            
            groups_df = pd.DataFrame(groups_data, columns=['Group_ID', 'Labels', 'Total_Count', 'Proposed_Canonical'])
            groups_df.to_excel(writer, sheet_name='Similar_Groups', index=False)
            
            # Sheet 3: Detailed Analysis
            analysis_data = analyze_similar_groups(similar_groups, edge_counts)
            analysis_df = pd.DataFrame(analysis_data)
            analysis_df.to_excel(writer, sheet_name='Detailed_Analysis', index=False)
            
            # Sheet 4: Summary statistics
            summary_data = [
                ['Total_Relationship_Types', len(edge_counts)],
                ['Total_Relationships', sum(edge_counts.values())],
                ['Similar_Groups_Found', len(similar_groups)],
                ['', ''],  # Empty row
                ['Top_10_Relationship_Types', '']
            ]
            
            for i, (label, count) in enumerate(sorted_edges[:10], 1):
                summary_data.append([f'  {i}. {label}', count])
            
            summary_df = pd.DataFrame(summary_data, columns=['Metric', 'Value'])
            summary_df.to_excel(writer, sheet_name='Summary', index=False)
        else:
            # Create empty similar groups sheet
            empty_groups_df = pd.DataFrame(columns=['Group_ID', 'Labels', 'Total_Count', 'Proposed_Canonical'])
            empty_groups_df.to_excel(writer, sheet_name='Similar_Groups', index=False)
            
            # Create empty detailed analysis sheet
            empty_analysis_df = pd.DataFrame(columns=['Group_ID', 'Labels', 'Total_Count', 'Proposed_Canonical', 'Analysis', 'Recommendation'])
            empty_analysis_df.to_excel(writer, sheet_name='Detailed_Analysis', index=False)
            
            # Sheet 3: Summary statistics
            summary_data = [
                ['Total_Relationship_Types', len(edge_counts)],
                ['Total_Relationships', sum(edge_counts.values())],
                ['Similar_Groups_Found', 0],
                ['', ''],  # Empty row
                ['Top_10_Relationship_Types', '']
            ]
            
            for i, (label, count) in enumerate(sorted_edges[:10], 1):
                summary_data.append([f'  {i}. {label}', count])
            
            summary_df = pd.DataFrame(summary_data, columns=['Metric', 'Value'])
            summary_df.to_excel(writer, sheet_name='Summary', index=False)
    
    print(f"Created Excel file: {excel_path}")
    return excel_path

def analyze_kg_duplicates(kg_dir: str, kg_type: str, output_dir: str):
    """
    Analyze a knowledge graph directory for relationship type duplicates
    """
    graph_store_path = os.path.join(kg_dir, 'graph_store.json')
    
    if not os.path.exists(graph_store_path):
        print(f"Error: Graph store not found in {kg_dir}")
        return
    
    print(f"\n{'='*60}")
    print(f"Analyzing {kg_type.upper()} Knowledge Graph for Duplicates:")
    print(f"{'='*60}")
    
    # Extract relationship types
    edge_counts = extract_relationship_types(graph_store_path)
    
    if not edge_counts:
        print(f"No relationship types found in {kg_type}")
        return
    
    # Generate Excel file
    excel_path = generate_excel_file(kg_type, edge_counts, output_dir)
    
    # Print top relationship types
    print(f"\nTop 20 Relationship Types in {kg_type}:")
    sorted_edges = sorted(edge_counts.items(), key=lambda x: x[1], reverse=True)
    for i, (label, count) in enumerate(sorted_edges[:20], 1):
        print(f"{i:2d}. {label:<30} ({count:>6} times)")
    
    # Find and display similar groups
    similar_groups = find_similar_labels(edge_counts)
    if similar_groups:
        print(f"\nFound {len(similar_groups)} groups of similar labels:")
        for i, group in enumerate(similar_groups, 1):
            total_count = sum(edge_counts[label] for label in group)
            print(f"Group {i}: {group} (Total: {total_count})")
    else:
        print("\nNo obvious duplicate groups found automatically.")

def main():
    """
    Main function to analyze all knowledge graphs for duplicates
    """
    base_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'local_kg_storage')
    kg_types = ['solidity_original_backup_v2']
    
    # Create output directory
    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'duplicates_analysis_solidity_test')
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Base directory: {base_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Analyzing knowledge graphs: {kg_types}")
    
    for kg_type in kg_types:
        kg_dir = os.path.join(base_dir, kg_type)
        
        if not os.path.exists(kg_dir):
            print(f"Error: Directory does not exist: {kg_dir}")
            continue
            
        analyze_kg_duplicates(kg_dir, kg_type, output_dir)
    
    print(f"\n{'='*60}")
    print("DUPLICATES ANALYSIS COMPLETE")
    print(f"{'='*60}")
    print(f"All Excel files have been generated in: {output_dir}")
    print("\nFiles created for each KG:")
    print("- {kg_type}_duplicates_analysis.xlsx - Excel file with 4 sheets:")
    print("  * Sheet 1: Duplicates - Main duplicates sheet for manual review")
    print("  * Sheet 2: Similar_Groups - Automatically detected similar groups")
    print("  * Sheet 3: Detailed_Analysis - Analysis and recommendations for each group")
    print("  * Sheet 4: Summary - Summary statistics")
    print("\nNext steps:")
    print("1. Open the Excel files and review the Duplicates sheet")
    print("2. Check the Detailed_Analysis sheet for automatic suggestions")
    print("3. Identify obvious duplicates and similar relationship types")
    print("4. Propose canonical names for each group")
    print("5. Document merge rules for implementation")

if __name__ == "__main__":
    main() 
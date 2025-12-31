import urllib.request
import urllib.parse
import xml.etree.ElementTree as ET
from datetime import datetime
from collections import defaultdict
import csv
import time
import json
import matplotlib.pyplot as plt
import os

def fetch_arxiv_papers_batch(search_query, max_results=1000, start=0, search_field='ti', exact_phrase=False):
    """
    Fetch a single batch of papers from arXiv API.
    
    Args:
        search_query: Search terms (e.g., "continual reinforcement learning")
        max_results: Maximum number of results to fetch in this batch
        start: Starting index for pagination
        search_field: Field to search in:
            'ti' = title only
            'abs' = abstract only
            'ti_or_abs' = title OR abstract (broad - finds papers with phrase in either)
            'ti_and_abs' = title AND abstract (strict - phrase must be in both)
            'all' = all fields (title, abstract, authors, comments, etc.)
        exact_phrase: If True, search for exact phrase; if False, search for all words
    
    Returns:
        List of paper dictionaries
    """
    base_url = 'http://export.arxiv.org/api/query?'
    
    # For exact phrase, wrap in quotes
    if exact_phrase:
        search_term = f'"{search_query}"'
    else:
        # For non-exact phrase, we need ALL words present (but in any order)
        # We'll construct AND conditions for each word
        words = search_query.split()
        search_term = words  # Keep as list for special handling
    
    # Build the query based on search field
    if search_field == 'ti_or_abs':
        if exact_phrase:
            query_string = f'ti:{search_term} OR abs:{search_term}'
        else:
            # For non-exact: (word1 in ti OR abs) AND (word2 in ti OR abs) AND (word3 in ti OR abs)
            word_conditions = []
            for word in search_term:
                word_conditions.append(f'(ti:{word} OR abs:{word})')
            query_string = ' AND '.join(word_conditions)
    elif search_field == 'ti_and_abs':
        if exact_phrase:
            query_string = f'ti:{search_term} AND abs:{search_term}'
        else:
            # All words must appear in title AND all words must appear in abstract
            ti_conditions = ' AND '.join([f'ti:{word}' for word in search_term])
            abs_conditions = ' AND '.join([f'abs:{word}' for word in search_term])
            query_string = f'({ti_conditions}) AND ({abs_conditions})'
    else:
        # Single field (ti, abs, or all)
        if exact_phrase:
            query_string = f'{search_field}:{search_term}'
        else:
            # All words must appear in the specified field
            word_conditions = [f'{search_field}:{word}' for word in search_term]
            query_string = ' AND '.join(word_conditions)
    
    query_params = {
        'search_query': query_string,
        'start': start,
        'max_results': max_results,
        'sortBy': 'submittedDate',
        'sortOrder': 'descending'
    }
    
    url = base_url + urllib.parse.urlencode(query_params)
    
    # Debug: print the actual URL being queried (first batch only)
    if start == 0:
        print(f"  Query URL: {url}\n")
    
    try:
        with urllib.request.urlopen(url) as response:
            data = response.read().decode('utf-8')
        
        # Parse XML
        root = ET.fromstring(data)
        
        # Namespace for arXiv API
        ns = {
            'atom': 'http://www.w3.org/2005/Atom',
            'arxiv': 'http://arxiv.org/schemas/atom'
        }
        
        papers = []
        entries = root.findall('atom:entry', ns)
        
        for entry in entries:
            title = entry.find('atom:title', ns).text.strip().replace('\n', ' ')
            published = entry.find('atom:published', ns).text
            arxiv_id = entry.find('atom:id', ns).text.split('/abs/')[-1]
            
            # Get authors
            authors = []
            for author in entry.findall('atom:author', ns):
                name = author.find('atom:name', ns).text
                authors.append(name)
            
            # Get summary/abstract
            summary = entry.find('atom:summary', ns).text.strip().replace('\n', ' ')
            
            # Parse date
            pub_date = datetime.strptime(published, '%Y-%m-%dT%H:%M:%SZ')
            
            papers.append({
                'title': title,
                'arxiv_id': arxiv_id,
                'published': pub_date,
                'year': pub_date.year,
                'month': pub_date.month,
                'authors': authors,
                'summary': summary
            })
        
        return papers
    
    except Exception as e:
        print(f"Error fetching batch starting at {start}: {e}")
        return []

def filter_papers_locally(papers, filter_terms, exact_phrase=False, case_sensitive=False):
    """
    Apply additional local filtering to papers after fetching.
    
    Args:
        papers: List of paper dictionaries
        filter_terms: Terms to filter for (e.g., "continual reinforcement learning")
        exact_phrase: If True, match exact phrase; if False, match all words
        case_sensitive: If True, case-sensitive matching
    
    Returns:
        Filtered list of papers
    """
    if not filter_terms:
        return papers
    
    filtered = []
    
    for paper in papers:
        # Combine title and abstract for searching
        search_text = f"{paper['title']} {paper['summary']}"
        
        if not case_sensitive:
            search_text = search_text.lower()
            filter_terms_to_use = filter_terms.lower()
        else:
            filter_terms_to_use = filter_terms
        
        # Check if the terms match
        if exact_phrase:
            if filter_terms_to_use in search_text:
                filtered.append(paper)
        else:
            # Check if all words are present
            words = filter_terms_to_use.split()
            if all(word in search_text for word in words):
                filtered.append(paper)
    
    return filtered

def fetch_arxiv_papers(search_query, max_results=1000, batch_size=1000, search_field='ti', 
                       exact_phrase=False, local_filter=None):
    """
    Fetch papers from arXiv API with pagination support.
    
    Args:
        search_query: Search terms (e.g., "continual reinforcement learning")
        max_results: Total maximum number of results to fetch
        batch_size: Number of results per API request (max ~1000-2000)
        search_field: Field to search in ('ti', 'abs', 'ti_or_abs', 'all')
        exact_phrase: If True, search for exact phrase in arXiv
        local_filter: If provided, apply additional local filtering after fetch
    
    Returns:
        Tuple of (list of paper dictionaries, boolean indicating if we hit max_results)
    """
    print(f"Fetching papers from arXiv...")
    print(f"Search query: '{search_query}'")
    print(f"Search field: {search_field}")
    print(f"Exact phrase: {exact_phrase}")
    print(f"Target: {max_results} papers (in batches of {batch_size})")
    print()
    
    all_papers = []
    seen_ids = set()  # Track arXiv IDs to detect duplicates
    start = 0
    hit_max_results = False
    
    # For EXACT_PHRASE=False, we'll do local verification to ensure all words are present
    verify_all_words = not exact_phrase and local_filter is None
    
    while start < max_results:
        # Calculate how many to fetch in this batch
        current_batch_size = min(batch_size, max_results - start)
        
        print(f"Fetching batch: papers {start+1} to {start+current_batch_size}...")
        
        papers = fetch_arxiv_papers_batch(search_query, current_batch_size, start, search_field, exact_phrase)
        
        if not papers:
            print(f"No more papers found or error occurred at offset {start}")
            break
        
        # Filter out duplicates (can happen if papers are added during fetching)
        new_papers = []
        duplicates = 0
        filtered_missing_words = 0
        
        for paper in papers:
            if paper['arxiv_id'] not in seen_ids:
                # Check if all words are present (for EXACT_PHRASE=False)
                if verify_all_words:
                    search_text = f"{paper['title']} {paper['summary']}".lower()
                    if search_field == 'ti':
                        search_text = paper['title'].lower()
                    elif search_field == 'abs':
                        search_text = paper['summary'].lower()
                    
                    words = search_query.lower().split()
                    if all(word in search_text for word in words):
                        seen_ids.add(paper['arxiv_id'])
                        new_papers.append(paper)
                    else:
                        filtered_missing_words += 1
                else:
                    seen_ids.add(paper['arxiv_id'])
                    new_papers.append(paper)
            else:
                duplicates += 1
        
        if duplicates > 0:
            print(f"  Warning: Found {duplicates} duplicate(s) - filtered out")
        if filtered_missing_words > 0:
            print(f"  Warning: Filtered {filtered_missing_words} paper(s) missing required words")
        
        all_papers.extend(new_papers)
        print(f"  Retrieved {len(new_papers)} valid papers (Total so far: {len(all_papers)})")
        
        # If we got fewer papers than requested, we've reached the end
        if len(papers) < current_batch_size:
            print(f"Reached end of results (got {len(papers)} papers, expected {current_batch_size})")
            break
        
        start += len(papers)
        
        # Check if we're about to hit max_results
        if start >= max_results:
            hit_max_results = True
        
        # Be nice to the arXiv API - add a delay between requests
        if start < max_results:
            print("  Waiting 3 seconds before next request...")
            time.sleep(3)
    
    print(f"\nTotal papers fetched from arXiv: {len(all_papers)}")
    if hit_max_results:
        print(f"Note: Hit MAX_RESULTS limit ({max_results}). There may be more papers available.")
    else:
        print("Note: Fetched all available papers matching the query.")
    
    # Apply local filtering if specified
    if local_filter:
        print(f"\nApplying local filter: '{local_filter}'")
        all_papers = filter_papers_locally(all_papers, local_filter, exact_phrase=True)
        print(f"Papers after local filtering: {len(all_papers)}")
    
    return all_papers, hit_max_results

def analyze_papers(papers):
    """Analyze papers and generate statistics."""
    
    # Count by year
    year_counts = defaultdict(int)
    for paper in papers:
        year_counts[paper['year']] += 1
    
    # Count by month
    month_counts = defaultdict(int)
    for paper in papers:
        month_key = f"{paper['year']}-{paper['month']:02d}"
        month_counts[month_key] += 1
    
    # Sort
    year_data = sorted(year_counts.items())
    month_data = sorted(month_counts.items())
    
    return {
        'total': len(papers),
        'year_counts': dict(year_data),
        'month_counts': dict(month_data),
        'years': [y for y, _ in year_data],
        'year_values': [c for _, c in year_data],
        'papers': papers
    }

def save_to_csv(data, filename_prefix='crl_publications', output_dir='results'):
    """Save data to CSV files."""
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Save year counts
    year_file = os.path.join(output_dir, f'{filename_prefix}_by_year.csv')
    with open(year_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['Year', 'Count'])
        for year, count in sorted(data['year_counts'].items()):
            writer.writerow([year, count])
    print(f"Saved year data to {year_file}")
    
    # Save month counts
    month_file = os.path.join(output_dir, f'{filename_prefix}_by_month.csv')
    with open(month_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['Month', 'Count'])
        for month, count in sorted(data['month_counts'].items()):
            writer.writerow([month, count])
    print(f"Saved month data to {month_file}")
    
    # Save full paper list
    papers_file = os.path.join(output_dir, f'{filename_prefix}_full_list.csv')
    with open(papers_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['Title', 'ArXiv ID', 'Year', 'Month', 'Published Date', 'Authors', 'Abstract'])
        for paper in data['papers']:
            writer.writerow([
                paper['title'],
                paper['arxiv_id'],
                paper['year'],
                paper['month'],
                paper['published'].strftime('%Y-%m-%d'),
                '; '.join(paper['authors']),
                paper['summary']
            ])
    print(f"Saved full paper list to {papers_file}")
    
    # Save summary statistics
    stats_file = os.path.join(output_dir, f'{filename_prefix}_statistics.json')
    with open(stats_file, 'w', encoding='utf-8') as f:
        stats = {
            'total_papers': data['total'],
            'year_range': f"{min(data['year_counts'].keys())} - {max(data['year_counts'].keys())}",
            'peak_year': max(data['year_counts'].items(), key=lambda x: x[1]),
            'year_counts': data['year_counts'],
            'month_counts': data['month_counts']
        }
        json.dump(stats, f, indent=2)
    print(f"Saved statistics to {stats_file}")

def print_summary(data):
    """Print summary statistics."""
    print("\n" + "="*60)
    print("SUMMARY STATISTICS")
    print("="*60)
    print(f"Total papers found: {data['total']}")
    print(f"Year range: {min(data['year_counts'].keys())} - {max(data['year_counts'].keys())}")
    
    peak_year, peak_count = max(data['year_counts'].items(), key=lambda x: x[1])
    print(f"Peak year: {peak_year} ({peak_count} papers)")
    
    print("\nPublications by year:")
    for year, count in sorted(data['year_counts'].items()):
        bar = '█' * (count // 2)  # Scale for display
        print(f"  {year}: {count:3d} {bar}")
    
    # Calculate growth rates
    print("\nYear-over-year growth:")
    years = sorted(data['year_counts'].keys())
    for i in range(1, len(years)):
        prev_year = years[i-1]
        curr_year = years[i]
        prev_count = data['year_counts'][prev_year]
        curr_count = data['year_counts'][curr_year]
        growth = ((curr_count - prev_count) / prev_count) * 100
        print(f"  {curr_year}: {growth:+.1f}%")

def plot_publications(data, filename_prefix='crl_publications', output_dir='results', 
                     drop_earliest=True):
    """Create a bar chart of publications over time."""
    
    
    # Set plots to match ICML text
    plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "STIXGeneral"],
    "mathtext.fontset": "stix",         # matches Times-like math
    "axes.titlesize": 12,
    "axes.labelsize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 8
    })
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    years = sorted(data['year_counts'].keys())
    counts = [data['year_counts'][year] for year in years]
    
    # Remove the earliest year (likely incomplete data) only if requested
    if drop_earliest and len(years) > 1:
        earliest_year = years[0]
        earliest_count = counts[0]
        print(f"\nNote: Removing earliest year ({earliest_year}) from plot (had {earliest_count} papers)")
        print(f"      This year likely has incomplete data from the search.")
        years = years[1:]
        counts = counts[1:]
    elif not drop_earliest:
        print(f"\nNote: Keeping earliest year in plot (fetched all available papers)")
    
    # Create figure with higher DPI for publication quality
    fig, ax = plt.subplots(figsize=(4, 3), dpi=300)
    
    # Create the bar chart
    ax.bar(years, counts, color='#2563eb', edgecolor='none')
    
    # Customize the plot
    ax.set_xlabel('Year')
    ax.set_ylabel('Paper Count')
    
    # No gridlines
    ax.grid(False)
    
    # Ensure all years are shown on x-axis
    ax.set_xticks(years)
    ax.set_xticklabels(years, rotation=45, ha='right')
    
    # Set y-axis limits - start from slightly below minimum to emphasize trend
    y_min = min(counts)
    y_max = max(counts)
    y_range = y_max - y_min
    ax.set_ylim(max(0, y_min - y_range * 0.15), y_max * 1.05)
    
    # Tight layout to prevent label cutoff
    plt.tight_layout()
    
    # Save the plot
    plot_file = os.path.join(output_dir, f'{filename_prefix}_plot.png')
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    print(f"Saved plot to {plot_file}")
    
    # Also save as PDF for LaTeX papers
    plot_file_pdf = os.path.join(output_dir, f'{filename_prefix}_plot.pdf')
    plt.savefig(plot_file_pdf, format='pdf', bbox_inches='tight')
    print(f"Saved plot to {plot_file_pdf}")
    
    # Close the figure to free memory
    plt.close()
    
    print("Plot generation complete!")

def main():
    # Configuration
    SEARCH_QUERY = "continual reinforcement learning"
    MAX_RESULTS = 100000  # Can now fetch more than 1000!
    BATCH_SIZE = 1000   # Number of papers per API request
    OUTPUT_DIR = 'results'  # Directory to save all output files
    
    # Search field options:
    # 'ti' = title only (most strict - only papers with phrase in title)
    # 'abs' = abstract only (papers with phrase in abstract)
    # 'ti_or_abs' = title OR abstract (broad - phrase in either title or abstract)
    # 'ti_and_abs' = title AND abstract (very strict - phrase must be in BOTH)
    # 'all' = all fields (broadest, includes everything)
    SEARCH_FIELD = 'ti_or_abs'

    # Exact phrase matching:
    # True = search for exact phrase "continual reinforcement learning"
    # False = search for papers containing all words (in any order)
    EXACT_PHRASE = False
    
    # Local filtering (optional):
    # If you want to fetch broadly then filter locally, set this
    # For example, fetch "reinforcement learning" then filter for "continual"
    LOCAL_FILTER = None  # Set to None to disable, or e.g., "continual reinforcement learning"
    
    print("="*60)
    print("arXiv Continual RL Publication Analyzer")
    print("="*60)
    print(f"Search query: '{SEARCH_QUERY}'")
    print(f"Search field: {SEARCH_FIELD}")
    print(f"Exact phrase: {EXACT_PHRASE}")
    print(f"Max results: {MAX_RESULTS}")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Output directory: {OUTPUT_DIR}")
    if LOCAL_FILTER:
        print(f"Local filter: '{LOCAL_FILTER}'")
    print()
    
    print("CONFIGURATION OPTIONS:")
    print("  SEARCH_FIELD:")
    print("    'ti' = title only (strictest, recommended)")
    print("    'abs' = abstract only")
    print("    'ti_or_abs' = title OR abstract (broader)")
    print("    'ti_and_abs' = title AND abstract (very strict)")
    print("    'all' = all fields (broadest)")
    print("  EXACT_PHRASE: True (exact phrase), False (all words)")
    print("  LOCAL_FILTER: Apply additional filtering after fetching")
    print()
    
    # Fetch papers
    papers, hit_max_results = fetch_arxiv_papers(SEARCH_QUERY, MAX_RESULTS, BATCH_SIZE, SEARCH_FIELD, 
                                EXACT_PHRASE, LOCAL_FILTER)
    
    if not papers:
        print("No papers found or error occurred.")
        return
    
    # Analyze
    print("\nAnalyzing papers...")
    data = analyze_papers(papers)
    
    # Print summary
    print_summary(data)
    
    # Save to files
    print("\n" + "="*60)
    print("SAVING DATA")
    print("="*60)
    save_to_csv(data, output_dir=OUTPUT_DIR)
    
    # Create plot
    print("\n" + "="*60)
    print("CREATING PLOT")
    print("="*60)
    # Only drop earliest year if we hit the max_results limit (incomplete data)
    plot_publications(data, output_dir=OUTPUT_DIR, drop_earliest=hit_max_results)
    
    print("\n" + "="*60)
    print("DONE!")
    print("="*60)
    print(f"\nAll files saved to '{OUTPUT_DIR}/' directory")
    print("You can now use the CSV files and plots in your paper.")
    print("Consider creating visualizations with matplotlib or seaborn.")

if __name__ == "__main__":
    main()
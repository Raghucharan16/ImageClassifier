import os
import json
import shutil
import argparse
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def segregate_images(processed_dir, results_json, output_dir):
    """
    Reads the classification results JSON and copies images from processed_dir
    into respective subdirectories inside output_dir based on classification labels.
    """
    logging.info(f"Starting image segregation using {results_json}...")
    
    # 1. Load results JSON
    if not os.path.exists(results_json):
        logging.error(f"Results file {results_json} not found. Please run classification first.")
        return
        
    with open(results_json, "r", encoding="utf-8") as f:
        results = json.load(f)
        
    logging.info(f"Loaded {len(results)} classification entries from JSON.")
    
    # 2. Valid folders come from the classifier taxonomy (+ any labels present).
    from rules import CATEGORIES
    categories = list(CATEGORIES)
    for cat in {info.get("recommended_folder", "unidentified") for info in results.values()}:
        if cat not in categories:
            categories.append(cat)

    # Create output directories
    for cat in categories:
        cat_path = os.path.join(output_dir, cat)
        os.makedirs(cat_path, exist_ok=True)
        
    # 3. Segregate files
    copied_counts = {cat: 0 for cat in categories}
    total_files = len(results)
    
    for filename, info in results.items():
        src_path = os.path.join(processed_dir, filename)
        
        # If the file doesn't exist in processed_dir, log error and skip
        if not os.path.exists(src_path):
            logging.warning(f"File {filename} found in JSON but not in processed directory: {src_path}")
            continue
            
        recommended_folder = info.get("recommended_folder", "unidentified")
        
        # Validate category
        if recommended_folder not in categories:
            logging.warning(f"Invalid category '{recommended_folder}' for file {filename}. Moving to 'unidentified'.")
            recommended_folder = "unidentified"
            
        dest_path = os.path.join(output_dir, recommended_folder, filename)
        
        try:
            shutil.copy2(src_path, dest_path)
            copied_counts[recommended_folder] += 1
        except Exception as e:
            logging.error(f"Failed to copy {filename} to {recommended_folder}: {e}")
            
    # 4. Report summary
    logging.info("========================================")
    logging.info("Segregation Summary:")
    logging.info(f"Total processed files: {total_files}")
    for cat, count in copied_counts.items():
        logging.info(f"  - {cat}: {count} files")
    logging.info("========================================")
    logging.info(f"All images have been successfully copied into {output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Segregate Preprocessed Images based on Classification JSON")
    parser.add_argument("--processed-dir", default="processed", help="Path to preprocessed images folder")
    parser.add_argument("--results-json", default="classification_results.json", help="Path to classification JSON file")
    parser.add_argument("--output-dir", default="segregated", help="Path to target directory for segregation subfolders")
    
    args = parser.parse_args()
    segregate_images(args.processed_dir, args.results_json, args.output_dir)

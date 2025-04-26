import os

output_ctm_path = "WORK_DIR/nfa_output/ctm/"

# Read and display CTM files
print(f"{'Word':<20}{'Start Time':<15}{'Duration':<10}")
print("=" * 50)

for subdir in ["words", "tokens", "segments"]:
    subdir_path = os.path.join(output_ctm_path, subdir)
    if os.path.isdir(subdir_path):
        for ctm_file in os.listdir(subdir_path):
            ctm_file_path = os.path.join(subdir_path, ctm_file)
            if os.path.isfile(ctm_file_path):
                print(f"\nReading file: {ctm_file}")
                with open(ctm_file_path, "r") as f:
                    for line in f.readlines():
                        parts = line.strip().split()
                        if len(parts) >= 4:  # Ensure there's enough data
                            word = parts[4]  # Adjust index if needed
                            start_time = parts[2]
                            duration = parts[3]
                            print(f"{word:<20}{start_time:<15}{duration:<10}")
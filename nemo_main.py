import os
import time
import json
import traceback
import re
import argparse
import ulid
import subprocess

class TextReader:
    def processMarks(self, text, file_name, audio_path):
        start_time = time.time()
        try:
            response = {}
            root_path = os.path.dirname(os.path.abspath(__file__)) + "/../storage/app/texthighlights/"
            os.makedirs(root_path, exist_ok=True)  # Ensure directory exists

            # Check if the audio file exists
            if not os.path.exists(audio_path):
                raise FileNotFoundError(f"Audio file not found: {audio_path}")

            # Preprocess the text
            text = re.sub(r'\n{3,}', '/nn/nn/nn', text)  # Handle triple newlines
            text = re.sub(r'\n{2}', '/nn/nn', text)      # Handle double newlines
            text = re.sub(r'\n', '/nn', text)           # Handle single newlines
            text = re.sub('\s+', ' ', text)  # Replace multiple spaces with a single space
            text = text.replace('<p>', '\n\n')  # Replace <p> with double newlines

            # Write the processed text to a manifest file
            manifest_data = {
                "audio_filepath": audio_path,
                "text": text,
            }
            manifest_file_path = os.path.join(root_path, f"{file_name}_manifest.json")
            with open(manifest_file_path, "w") as manifest_file:
                json.dump(manifest_data, manifest_file)

            # Perform forced alignment using NeMo
            nemo_output_dir = os.path.join(root_path, f"{file_name}_nfa_output")
            os.makedirs(nemo_output_dir, exist_ok=True)
            command = (
                f"python NeMo/tools/nemo_forced_aligner/align.py "
                f'pretrained_name="stt_en_fastconformer_hybrid_large_pc" '
                f'manifest_filepath="{manifest_file_path}" '
                f'output_dir="{nemo_output_dir}" '
                f'additional_segment_grouping_separator="|" '
                f'ass_file_config.vertical_alignment="bottom" '
                f'ass_file_config.text_already_spoken_rgb=[66,245,212] '
                f'ass_file_config.text_being_spoken_rgb=[242,222,44] '
                f'ass_file_config.text_not_yet_spoken_rgb=[223,242,239]'
            )
            result = subprocess.run(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            if result.returncode != 0:
                raise RuntimeError(f"NeMo aligner failed: {result.stderr.decode('utf-8')}")

            # Read the generated CTM file for word alignments
            ctm_dir = os.path.join(nemo_output_dir, "ctm", "words")
            response['marks'] = []
            if os.path.exists(ctm_dir):
                for ctm_file in os.listdir(ctm_dir):
                    ctm_file_path = os.path.join(ctm_dir, ctm_file)
                    if os.path.isfile(ctm_file_path):
                        with open(ctm_file_path, "r") as ctm:
                            for line in ctm.readlines():
                                parts = line.strip().split()
                                if len(parts) >= 4:  # Ensure sufficient data in the line
                                    word = parts[4]
                                    start_time = float(parts[2])
                                    duration = float(parts[3])
                                    if word != "<b>":
                                        word_cleaned = word.replace("▁", "")
                                        response['marks'].append({
                                            's': start_time,
                                            'e': start_time + duration,
                                            'w': word_cleaned
                                        })
            else:
                raise FileNotFoundError(f"CTM directory not found: {ctm_dir}")

            # Cleanup temporary files
            os.remove(manifest_file_path)
            response['status'] = True
        except Exception as e:
            # Log detailed error information
            traceback.print_exc()
            response['status'] = False
            response['message'] = f"{str(e)}\n{traceback.format_exc()}"
        response['time'] = float("{:.2f}".format(time.time() - start_time))
        return json.dumps(response)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process text highlight')
    parser.add_argument('--text', help='Input string', required=True)
    parser.add_argument('--audiopath', help='Audio file path', required=True)

    args = parser.parse_args()
    text = args.text
    file_name = str(ulid.ulid())
    audio_path = args.audiopath

    text_reader = TextReader()
    response = text_reader.processMarks(text, file_name, audio_path)
    print(response)

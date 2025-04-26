import os
import time
import json
import traceback
import re
import argparse
import ulid

class TextReader:
    def processMarks(self, text, file_name, audio_path):
        start_time = time.time()
        try:
            response = {}

            # Ensure root_path is absolute
            root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../storage/app/texthighlights/"))
            os.makedirs(root_path, exist_ok=True)  # Ensure directory exists

            # Check if the audio file exists
            if not os.path.exists(audio_path):
                raise FileNotFoundError(f"Audio file not found: {audio_path}")

            # Preprocess the text
            text = re.sub(r'\n{3,}', '/nn/nn/nn', text)
            text = re.sub(r'\n{2}', '/nn/nn', text)
            text = re.sub(r'\n', '/nn', text)
            text = re.sub(r'\s+', ' ', text)
            text = text.replace('<p>', '\n\n')

            # Write the processed text to a manifest file
            manifest_file_path = os.path.join(root_path, f"{file_name}_manifest.json")
            manifest_data = {"audio_filepath": audio_path, "text": text}
            with open(manifest_file_path, "w") as manifest_file:
                json.dump(manifest_data, manifest_file)

            # Perform forced alignment using NeMo
            nemo_output_dir = os.path.join(root_path, f"{file_name}_nfa_output")
            os.makedirs(nemo_output_dir, exist_ok=True)
            command = (
                f"python3 /var/www/html/core/NeMo/tools/nemo_forced_aligner/align.py "
                f'pretrained_name="stt_en_fastconformer_hybrid_large_pc" '
                f'manifest_filepath="{manifest_file_path}" '
                f'output_dir="{nemo_output_dir}" '
                f'additional_segment_grouping_separator="|" '
                f'ass_file_config.vertical_alignment="bottom" '
                f'ass_file_config.text_already_spoken_rgb=[66,245,212] '
                f'ass_file_config.text_being_spoken_rgb=[242,222,44] '
                f'ass_file_config.text_not_yet_spoken_rgb=[223,242,239]'
            )
            result = os.system(command)
            if result != 0:
                raise RuntimeError("NeMo forced aligner command failed.")

            # Read the generated CTM file for word alignments
            ctm_dir = os.path.join(nemo_output_dir, "ctm", "words")
            if not os.path.exists(ctm_dir):
                raise FileNotFoundError(f"CTM directory not found: {ctm_dir}")

            response['marks'] = []
            for ctm_file in os.listdir(ctm_dir):
                ctm_file_path = os.path.join(ctm_dir, ctm_file)
                if os.path.isfile(ctm_file_path):
                    with open(ctm_file_path, "r") as ctm:
                        for line in ctm.readlines():
                            parts = line.strip().split()
                            if len(parts) >= 4:
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

            # Cleanup temporary files
            os.remove(manifest_file_path)
            response['status'] = True
        except Exception as e:
            traceback.print_exc()
            response['status'] = False
            response['message'] = f"{str(e)}\n{traceback.format_exc()}"
        response['time'] = float("{:.2f}".format(time.time() - start_time))
        return json.dumps(response)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process text highlight')
    parser.add_argument('--text', help='input string', required=True)
    parser.add_argument('--audiopath', help='audio url', required=True)

    args = parser.parse_args()
    text = args.text
    file_name = str(ulid.ulid())
    audio_path = args.audiopath
    text_reader = TextReader()
    response = text_reader.processMarks(text, file_name, audio_path)
    print(response)

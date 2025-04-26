import os
import time
import json
import traceback
import re
import argparse
import ulid
import subprocess
from concurrent.futures import ThreadPoolExecutor

class TextReader:
    def processMarks(self, text, file_name, audio_path):
        start_time = time.time()
        try:
            response = {}
            root_path = os.path.dirname(os.path.abspath(__file__)) + "/../storage/app/texthighlights/"
            os.makedirs(root_path, exist_ok=True)

            if not os.path.exists(audio_path):
                raise FileNotFoundError(f"Audio file not found: {audio_path}")

            # Precompiled regex for text preprocessing
            newline_pattern = re.compile(r'\n{3,}')
            double_newline_pattern = re.compile(r'\n{2}')
            single_newline_pattern = re.compile(r'\n')
            space_pattern = re.compile(r'\s+')

            text = newline_pattern.sub('/nn/nn/nn', text)
            text = double_newline_pattern.sub('/nn/nn', text)
            text = single_newline_pattern.sub('/nn', text)
            text = space_pattern.sub(' ', text)
            text = text.replace('<p>', '\n\n')

            # Create in-memory manifest
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

            command = [
                "python", "NeMo/tools/nemo_forced_aligner/align.py",
                'pretrained_name="stt_en_fastconformer_hybrid_large_pc"',
                f'manifest_filepath="{manifest_file_path}"',
                f'output_dir="{nemo_output_dir}"',
                'additional_segment_grouping_separator="|"',
                'ass_file_config.vertical_alignment="bottom"',
                'ass_file_config.text_already_spoken_rgb=[66,245,212]',
                'ass_file_config.text_being_spoken_rgb=[242,222,44]',
                'ass_file_config.text_not_yet_spoken_rgb=[223,242,239]'
            ]
            subprocess.run(" ".join(command), shell=True, check=True)

            # Process CTM files
            ctm_dir = os.path.join(nemo_output_dir, "ctm", "words")
            response['marks'] = []

            def process_ctm_file(ctm_file):
                ctm_file_path = os.path.join(ctm_dir, ctm_file)
                marks = []
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
                                    marks.append({
                                        's': start_time,
                                        'e': start_time + duration,
                                        'w': word_cleaned
                                    })
                return marks

            # Use threading to speed up CTM file processing
            with ThreadPoolExecutor() as executor:
                results = list(executor.map(process_ctm_file, os.listdir(ctm_dir)))
                for result in results:
                    response['marks'].extend(result)

            # Cleanup
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

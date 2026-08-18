import re
import logging
from pathlib import Path


from _trash.config_b_stare import VMRK_DATA_PATH, VMRK_CLEANED_PATH


logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)

def preprocess_all_vmrk_files(
    input_folder: Path = VMRK_DATA_PATH,
    output_folder: Path = VMRK_CLEANED_PATH
):
    """
    Reads .vmrk files, categorizes blocks based on trigger content,
    cleans metadata markers, and re-maps stimulus codes for uniform analysis.
    """

    output_folder.mkdir(parents=True, exist_ok=True)


    files = [f for f in input_folder.iterdir() if f.suffix.lower() == '.vmrk']

    if not files:
        log.warning("No .vmrk files found in %s", input_folder)
        return

    log.info("Processing %d files...", len(files))

    for input_file_path in files:

        output_file_path = output_folder / input_file_path.name

        with open(input_file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()


        all_content = "".join(lines)
        if "S 11" in all_content or "S 12" in all_content:
            block_type = "A"
        elif re.search(r"S\s+[56]\b", all_content):
            block_type = "B"
        elif "S 15" in all_content:
            block_type = "C"
        else:
            block_type = "Unknown"

        header_lines = []
        marker_lines = []
        in_marker_infos = False


        for line in lines:
            if line.strip() == "[Marker Infos]":
                in_marker_infos = True
                header_lines.append(line)
                continue

            if not in_marker_infos:
                header_lines.append(line)
            else:
                if line.strip().startswith("Mk"):
                    marker_lines.append(line.strip())
                else:
                    header_lines.append(line)

        processed_markers = []
        is_first_new_segment = True


        for line in marker_lines:
            match = re.match(r"Mk\d+=(.*?),(.*?),(.*)", line)
            if not match:
                continue

            m_type, m_desc, m_rest = [g.strip() for g in match.groups()]


            if m_type.lower() == "response":
                continue


            if m_type.lower() == "new segment":
                if is_first_new_segment:
                    m_type, m_desc, is_first_new_segment = "block start", "S 99", False
                else:
                    continue


            if m_type.lower() == "stimulus":
                norm_desc = re.sub(r'\s+', ' ', m_desc)
                mapping = {"S 5": "S 21", "S 6": "S 22", "S 15": "S 31"}
                m_desc = mapping.get(norm_desc, m_desc)

            processed_markers.append((m_type, m_desc, m_rest))


        with open(output_file_path, 'w', encoding='utf-8') as f:
            for line in header_lines:
                f.write(line)
                if line.strip() == "[Marker Infos]":
                    f.write("; Each entry: Mk<Marker number>=<Type>,<Description>,<Position in data points>,\n")
                    f.write("; <Size in data points>, <Channel number (0 = marker is related to all channels)>\n")

            for idx, (m_type, m_desc, m_rest) in enumerate(processed_markers, start=1):
                f.write(f"Mk{idx}={m_type},{m_desc},{m_rest}\n")

    log.info("Preprocessing complete. Processed %d files.", len(files))

if __name__ == "__main__":
    preprocess_all_vmrk_files()
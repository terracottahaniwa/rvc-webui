import os
import sys
import argparse
from typing import *

from pydantic import BaseModel

# reluctantly, RVC modules will be imported later in main()
# due to argparse confused if import them here.
# because utils.py refer cmd_opts.py, then already used argparse in it.


def main():
    # directories paths
    SCRIPTS_DIR = os.path.dirname(__file__)
    ROOT_DIR = os.path.normpath(os.path.join(SCRIPTS_DIR, ".."))

    # argparse
    parser = argparse.ArgumentParser(
        description=""
        "this is a inferring script. "
        "see this source code for "
        "what keys and values are in the json file."
    )
    parser.add_argument(
        "json_path", type=str,
        help=""
        "path to the inferring configuration json file. "
        "if the file is not found, "
        "write the template json to the same path."
    )
    args = parser.parse_args()
    if "-h" in args:
        return

    # configuration keys and values
    class ConfigurationJson(BaseModel):
        model: str = ""
        speaker_id: int = 0
        source_audio: str = ""
        out_folder: str = ""
        transpose: int = 0
        pitch_extraction_algorithm: (
            Literal["dio", "harvest", "mangio-crepe", "crepe"]) = "crepe"
        embedder_model: (Literal[
            "auto", "hubert-base-japanese", "contentvec"]) = "contentvec"
        embedder_output_layer: Literal["auto", "9", "12"] = "auto"
        auto_load_index: bool = False
        faiss_index_file_path: str = ""
        retrieval_feature_ratio: float = 1.0
        f0_curve_file: str = ""

    # load json
    try:
        config = ConfigurationJson.parse_file(args.json_path)
        print(f"loaded json file:\n{config.json(indent=2)}")
    except FileNotFoundError as e:
        print(f"{e}\ntemplate json file exported to same path.")
        config = ConfigurationJson()
        with open(args.json_path, "w") as f:
            f.write(config.json(indent=2))
        return

    # into global variables
    code = []
    for key, value in config.dict().items():
        code.append(f"{key} = {repr(value)}")
    exec(";".join(code), globals())

    global model
    global speaker_id
    global source_audio
    global out_folder
    global transpose
    global pitch_extraction_algorithm
    global embedder_model
    global embedder_output_layer
    global auto_load_index
    global faiss_index_file_path
    global retrieval_feature_ratio
    global f0_curve_file

    # verify range
    assert 0 <= speaker_id
    assert -20 <= transpose <= 20
    assert 0.0 <= retrieval_feature_ratio <= 1.0

    # glob files
    assert source_audio != ""
    source_audio = os.path.normpath(source_audio)

    files = [source_audio]
    if "*" in source_audio:
        glob_path = source_audio
        files = glob.glob(glob_path, recursive=True)
    if os.path.isdir(source_audio):
        glob_path = os.path.join(source_audio, "**", "*.wav")
        files = glob.glob(glob_path, recursive=True)

    # import RVC modules
    sys.path.append(ROOT_DIR)
    from modules import models

    # load model
    assert model != ""
    models.load_model(model)

    # inferring
    out_folder = out_folder or models.AUDIO_OUT_DIR
    out_folder = os.path.normpath(out_folder)
    assert os.path.isdir(out_folder)

    print("Inferring...")
    try:
        for file in files:
            print(file)
            models.vc_model.single(
                sid=speaker_id,
                input_audio=file,
                embedder_model_name=embedder_model,
                embedding_output_layer=embedder_output_layer,
                f0_up_key=transpose,
                f0_file=f0_curve_file,
                f0_method=pitch_extraction_algorithm,
                auto_load_index=auto_load_index,
                faiss_index_file=faiss_index_file_path,
                index_rate=retrieval_feature_ratio,
                output_dir=out_folder
            )
        print("Success")
    except BaseException as e:
        print("Error: ", e)


if __name__ == "__main__":
    main()

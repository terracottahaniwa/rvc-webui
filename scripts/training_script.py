import os
import sys
import shutil
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
    MODELS_DIR = os.path.join(ROOT_DIR, "models")

    # argparse
    parser = argparse.ArgumentParser(
        description=""\
        "this is a command line traning script. "\
        "see this source code for "\
        "what keys and values are in the json file."
    )
    parser.add_argument(
        "json_path", type=str,
        help=""\
        "path to the model configuration json file. "\
        "if the file is not found, "\
        "write the template json to the same path."
    )
    parser.add_argument(
        "--index-only", "-i", action='store_true',
        help="set this option if you want train index only."
    )
    args = parser.parse_args()
    if "-h" in args:
        return

    # configuration keys and values
    class ConfigurationJson(BaseModel):
        model_name: str = ""
        ignore_cache: bool = False
        dataset_glob: str = ""
        recursive: bool = True
        multiple_speakers: bool = False
        speaker_id: int = 0
        model_version: Literal["v1", "v2"] = "v2"
        target_sampling_rate: Literal["32k", "40k", "48k"] = "40k"
        f0_model: Literal["Yes", "No"] = "Yes"
        using_phone_embedder: (
            Literal["hubert-base-japanese", "contentvec"]) = "contentvec"
        embedding_channels: Literal["256", "768"] = "768"
        embedding_output_layer: Literal["9", "12"] = "12"
        gpu_id: str = "0"
        number_of_cpu_processes: int = os.cpu_count()
        normalize_audio_volume_when_preprocess: Literal["Yes", "No"] = "Yes"
        pitch_extraction_algorithm: (
            Literal["dio", "harvest", "mangio-crepe", "crepe"]) = "crepe"
        batch_size: int = 4
        number_of_epochs: int = 30
        save_every_epoch: int = 10
        cache_batch: bool = True
        fp16: bool = True
        augment: bool = False
        augment_from_pretrain: bool = False
        pre_trained_generator_path_pth: str = "file is not prepared"
        speaker_info_path_npy: str = "file is not prepared"
        pre_trained_generator_path: str = (
            os.path.join(MODELS_DIR, "pretrained", "v2", "f0G40k.pth"))
        pre_trained_discriminator_path: str = (
            os.path.join(MODELS_DIR, "pretrained", "v2", "f0D40k.pth"))
        train_index: Literal["Yes", "No"] = "Yes"
        reduce_index_size_with_kmeans: Literal["Yes", "No"] = "No"
        maximum_index_size: int = 10000

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

    # assign train_index to a variable with a different name.
    # because name conflict with same named function include later.
    global train_index
    train_index_ = train_index

    # additional directories paths
    TRAINING_DIR = os.path.join(MODELS_DIR, "training", "models", model_name)
    MUTE_DIR = os.path.join(MODELS_DIR, "training", "mute")
    OUT_DIR = os.path.join(MODELS_DIR, "checkpoints")
    EMBEDDING_DIR = os.path.join(MODELS_DIR, "embeddings")

    # rmtree and makedir
    if os.path.exists(TRAINING_DIR) and ignore_cache:
        shutil.rmtree(TRAINING_DIR)
    os.makedirs(TRAINING_DIR, exist_ok=True)

    # import RVC modules
    sys.path.append(ROOT_DIR)
    from lib.rvc.train import (
        create_dataset_meta,
        glob_dataset,
        train_index,
        train_model
    )
    from lib.rvc.preprocessing import (
        extract_f0,
        extract_feature,
        split
    )
    from modules.models import get_embedder
    from modules.utils import load_config
    from modules.shared import device

    # show --index-only is set or not
    print("--index-only is {}".format(
        "set" if args.index_only else "not set"))

    # glob dataset
    datasets = glob_dataset(
        glob_str=dataset_glob,
        speaker_id=speaker_id,
        multiple_speakers=multiple_speakers,
        recursive=recursive
    )
    if len(datasets) == 0:
        raise Exception("No audio files found")

    # preprocess
    print("Preprocessing...")
    sampling_rate = int(target_sampling_rate[:-1] + "000")
    norm_audio_when_preprocess = (
        normalize_audio_volume_when_preprocess == "Yes")
    split.preprocess_audio(
        datasets=datasets,
        sampling_rate=sampling_rate,
        num_processes=number_of_cpu_processes,
        training_dir=TRAINING_DIR,
        is_normalize=norm_audio_when_preprocess,
        mute_wav_path=os.path.join(
            MUTE_DIR, "0_gt_wavs", f"mute{target_sampling_rate}.wav")
    )

    # extract f0
    f0 = f0_model == "Yes"
    if f0:
        print("Extracting f0...")
        extract_f0.run(
            training_dir=TRAINING_DIR,
            num_processes=number_of_cpu_processes,
            f0_method=pitch_extraction_algorithm
        )

    # extruct features
    print("Extracting features...")
    embedder = get_embedder(embedder_name=using_phone_embedder)
    embedder_filepath, _, embedder_load_from = embedder
    if embedder_load_from == "local":
        embedder_filepath = os.path.join(EMBEDDING_DIR, embedder_filepath)
    gpu_ids = [int(x.strip()) for x in gpu_id.split(",")] if gpu_id else []
    extract_feature.run(
        training_dir=TRAINING_DIR,
        embedder_path=embedder_filepath,
        embedder_load_from=embedder_load_from,
        embedding_channel=int(embedding_channels),
        embedding_output_layer=int(embedding_output_layer),
        gpu_ids=gpu_ids
    )

    # dataset meta
    create_dataset_meta(TRAINING_DIR, f0)

    # training
    if not args.index_only:
        print("Training model...")
        if not augment_from_pretrain:
            pre_trained_generator_path_pth = None
            speaker_info_path_npy = None
        train_config = load_config(
            version=model_version,
            training_dir=TRAINING_DIR,
            sample_rate=target_sampling_rate,
            emb_channels=int(embedding_channels),
            fp16=fp16
        )
        train_model(
            gpus=gpu_ids,
            config=train_config,
            training_dir=TRAINING_DIR,
            model_name=model_name,
            out_dir=OUT_DIR,
            # sample_rate argument is annotated as an int type,
            # but actually must be passed as str to work.
            sample_rate=target_sampling_rate,
            f0=f0,
            batch_size=batch_size,
            augment=augment,
            augment_path=pre_trained_generator_path_pth,
            speaker_info_path=speaker_info_path_npy,
            cache_batch=cache_batch,
            total_epoch=number_of_epochs,
            save_every_epoch=save_every_epoch,
            pretrain_g=pre_trained_generator_path,
            pretrain_d=pre_trained_discriminator_path,
            embedder_name=using_phone_embedder,
            embedding_output_layer=int(embedding_output_layer),
            save_only_last=False,
            device=None if len(gpu_ids) > 1 else device
        )

    # training index
    print("Training index...")
    run_train_index = args.index_only or train_index_ == "Yes"
    reduce_index_size = reduce_index_size_with_kmeans == "Yes"
    if run_train_index:
        global maximum_index_size
        if not reduce_index_size:
            maximum_index_size = None
        train_index(
            training_dir=TRAINING_DIR,
            model_name=model_name,
            out_dir=OUT_DIR,
            emb_ch=int(embedding_channels),
            num_cpu_process=number_of_cpu_processes,
            maximum_index_size=maximum_index_size
        )
    print("Training complete")


if __name__ == "__main__":
    main()

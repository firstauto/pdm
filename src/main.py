import os
import argparse
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from pyparsing import col
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from pathlib import Path
import multiprocessing
from datetime import datetime
from timeit import default_timer as timer
from torch.utils.tensorboard import SummaryWriter  # type: ignore

from plot_utils import anomaly_plot
from seq_data import *
from utils import *

# Disable file validation for PyDev debugger
os.environ["PYDEVD_DISABLE_FILE_VALIDATION"] = "1"

# Set the multiprocessing start method to 'spawn'
multiprocessing.set_start_method("spawn", force=True)

# Set the random seed to make the results reproducible
set_seeds(313)

# Set up the device-agnostic code
device = "cuda" if torch.cuda.is_available() else "cpu"
# Set the data to name model
now = datetime.now()
date_string = now.strftime("%Y_%m_%d_%H_%M")

if __name__ == "__main__":

    # Parse command-line arguments, Initialize the parser
    parser = argparse.ArgumentParser(
        description="provide conditions for training the model"
    )
    parser.add_argument(
        "--RAW_DIR",
        type=lambda p: Path(p),
        help="path to the raw directory to load .csv data",
    )
    parser.add_argument(
        "--MODEL_DIR",
        type=lambda p: Path(p),
        help="path to save the model after training",
    )
    parser.add_argument(
        "--MODEL_TYPE",
        type=str,
        help="model for training, i.e. inception unet or transformer etc.",
    )
    parser.add_argument(
        "--DATA_TYPE", type=str, help="data type for training, i.e. '2017' or '2016'"
    )
    parser.add_argument(
        "--CASE",
        type=str,
        choices=["train", "test", "tune"],
        help="the case, i.e. 'train', 'test'",
    )
    parser.add_argument(
        "--SYSTEM",
        type=str,
        choices=["gearbox", "generator", "transformer", "hydraulic"],
        help="the case, i.e. 'gearbox', 'generator' etc.",
    )
    parser.add_argument("--BATCH_SIZE", type=int, help="batch size for the data")
    parser.add_argument("--EPOCHS", type=int, help="number of epochs")
    parser.add_argument(
        "--LEARNING_RATE", type=float, help="learning rate for training"
    )
    parser.add_argument(
        "--LEARNING",
        type=str,
        choices=["sup", "unsup"],
        help="Type of learning, i.e. supervised or unsupervised",
    )
    # Model specific arguments
    parser.add_argument("--SEQ_LEN", type=int, help="sequence length for training")
    parser.add_argument("--HIDDEN_SIZE", type=int, help="hidden size for training")
    parser.add_argument("--ENC_LAYERS", type=int, help="number of layers for training")
    parser.add_argument("--DEC_LAYERS", type=int, help="number of layers for training")
    parser.add_argument("--NUM_HEADS", type=int, help="number of heads for training")
    parser.add_argument("--EMB_SIZE", type=int, help="embedding size for training")
    parser.add_argument("--VAL_SPLIT", type=float, help="Ratio of validation split")
    parser.add_argument("--DROPOUT", type=float, help="dropout for training")
    # Training specific arguments
    parser.add_argument(
        "--LOSS_FN",
        type=str,
        choices=["mse", "hybrid"],
        help="loss function for training, i.e. celoss",
    )
    parser.add_argument(
        "--OPTIMIZER",
        type=str,
        choices=["adam", "rmsprop", "sgd"],
        help="optimizer for training, i.e. 'adam','rmsprop','sgd'",
    )
    parser.add_argument(
        "--SCHEDULER",
        type=str,
        choices=["cosine", "cyclic", "plateau", "multistep"],
        help="scheduler for training, i.e. 'cosine', 'cycle', 'plateau'",
    )
    parser.add_argument(
        "--NUM_WORKERS", type=int, help="number of workers for training"
    )

    args = parser.parse_args()

    # # Example usage:
    # from models.inception_unet import InceptionUNet
    # from models.ad_tfm import AD_TFM
    # InceptionUNet = InceptionUNet(in_channels=512, input_size=28, emb_size=64, kernel_size=None, stride=None, dropout=None)
    # AD_TFM = AD_TFM(d_model=28, emb_size=32, nhead=4, seq_len=512, num_encoder_layers=4,
    #                   num_decoder_layers=4, dim_feedforward=512//2, activation=nn.ReLU(), dropout=0.1,
    #                   norm_first=False)
    # input_tensor = torch.rand(32, 512, 28)  # Example input tensor with batch size=32, sequence length=100, and d_model=512
    # output = AD_TFM(input_tensor, input_tensor)

    # Make directory if it doesn't exist
    os.makedirs(args.MODEL_DIR, exist_ok=True)

    if args.LOSS_FN == "mse":
        loss_fn = nn.MSELoss(reduction="mean").to(device)
    elif args.LOSS_FN == "hybrid":
        loss_fn = kl_mse_loss
    else:
        raise ValueError(f"Invalid loss function: {args.LOSS_FN}")

    if args.CASE == "train":

        train_loaders, val_loaders, _, total_time = data_process(
            args.RAW_DIR,
            args.DATA_TYPE,
            args.CASE,
            args.SEQ_LEN,
            args.BATCH_SIZE,
            args.SYSTEM,
            args.VAL_SPLIT,
        )

        batch_data = next(iter(val_loaders[0]))  # type: ignore
        model_t = select_model(
            model_type=args.MODEL_TYPE,
            input_size=batch_data[0].shape[-1],
            emb_size=args.EMB_SIZE,
            seq_len=args.SEQ_LEN,
            num_heads=args.NUM_HEADS,
            hidden_size=args.HIDDEN_SIZE,
            enc_layers=args.ENC_LAYERS,
            dec_layers=args.DEC_LAYERS,
            dropout=args.DROPOUT,
            batch_size=args.BATCH_SIZE,
            device=device,
        )
        model_t.apply(initialize_weights)

        print(
            f"{args.MODEL_TYPE}:",
            sum(p.numel() for p in model_t.parameters() if p.requires_grad),
        )

        if args.OPTIMIZER == "adam":
            optimizer = adam_optimizer(
                model=model_t,
                lr=args.LEARNING_RATE,
                betas=(0.9, 0.999),
                eps=1e-08,
                weight_decay=0,
                amsgrad=True,
            )
        elif args.OPTIMIZER == "rmsprop":
            optimizer = rmsprop_optimizer(
                model=model_t,
                lr=args.LEARNING_RATE,
                alpha=0.99,
                eps=1e-08,
                weight_decay=0.01,
                momentum=0.9,
                centered=False,
            )
        elif args.OPTIMIZER == "sgd":
            optimizer = sgd_optimizer(
                model=model_t,
                lr=args.LEARNING_RATE,
                momentum=0.9,
                dampening=0,
                weight_decay=0.01,
                nesterov=True,
            )
        else:
            raise ValueError(f"Invalid optimizer: {args.OPTIMIZER}")

        scheduler = scheduler_lr(
            type=args.SCHEDULER,
            optimizer=optimizer,
            max_lr=args.LEARNING_RATE,
            min_lr=args.LEARNING_RATE / 1000,
        )

        es = EarlyStopping(patience=15, min_delta=0.000001)  # type: ignore

        writer = SummaryWriter(
            comment=f"_model{args.MODEL_TYPE}_data_{args.DATA_TYPE}_seqlen{args.SEQ_LEN}"
        )
        # Start the timer
        start_time = timer()
        model_result = train_and_evaluate(
            model=model_t,
            train_loaders=train_loaders,
            val_loaders=val_loaders,
            criterions=loss_fn,
            optimizer=optimizer,
            learning=args.LEARNING,
            num_epochs=args.EPOCHS,
            device=device,
            scheduler=scheduler,
            scheduler_type=args.SCHEDULER,
            es=es,
            tensorboard_writer=writer,
        )

        # End the timer and print out how long it took
        end_time = timer()
        print(f"[INFO] Total training time: {end_time - start_time:.3f} seconds")

        save_name = (
            f"{date_string}_model{args.MODEL_TYPE}_data{args.DATA_TYPE}_system{args.SYSTEM}_valsplit{args.VAL_SPLIT}"
            f"_batchsize{args.BATCH_SIZE}_seqlen{args.SEQ_LEN}_hiddensize{args.HIDDEN_SIZE}"
            f"_enclayers{args.ENC_LAYERS}_declayers{args.DEC_LAYERS}_embsize{args.EMB_SIZE}"
            f"_numheads{args.NUM_HEADS}_lrate{args.LEARNING_RATE}_dropout{args.DROPOUT}.pth"
        )

        # Save the model
        save_model(
            model=model_t,
            results=model_result,
            target_dir=f"{args.MODEL_DIR}/{args.DATA_TYPE}/",
            model_name=save_name,
            time=end_time - start_time,
        )

        # save models_results file as csv
        df = pd.DataFrame(data=model_result)
        _dir = f"{str(args.RAW_DIR)[:-4]}/results/logs/"
        os.makedirs(_dir, exist_ok=True)
        df.to_csv(f"{_dir}{save_name[:-4]}.csv", index=False)

    elif args.CASE == "test":

        tm = "latest"
        model_dir = f"{args.MODEL_DIR}/{args.DATA_TYPE}/"
        model_names = list(Path(model_dir).glob("*.pth"))
        model_names.sort(key=lambda x: x.stat().st_mtime, reverse=True)

        results = {
            "batch_size": [],
            "seq_len": [],
            "hidden_size": [],
            "layers": [],
            "n_heads": [],
            "l_rate": [],
            "dropout": [],
            "RMSE": [],
            "MAE": [],
        }
        for model_name in model_names:
            model_name = model_name.name
            model_parameters = model_name.split("_")
            model = model_parameters[5][5:]
            data_type = model_parameters[6][4:]
            system = model_parameters[7][6:]
            l_rate = model_parameters[-2][5:]
            dropout = model_parameters[-1][7:-4]
            val_split, b_size, s_len, h_size, n_layers, s_layers, e_size, n_heads = (
                extracting_parameters(model_parameters[8:-2])
            )

            test_loaders, train_loaders, test_time, train_time = data_process(
                args.RAW_DIR, data_type, args.CASE, s_len, b_size, system, val_split
            )

            batch_data = next(iter(test_loaders[0]))  # type: ignore
            model_t = select_model(
                model_type=model,
                input_size=batch_data[0].shape[-1],
                emb_size=e_size,
                seq_len=s_len,  # type: ignore
                hidden_size=h_size,
                enc_layers=n_layers,
                dec_layers=s_layers,
                num_heads=n_heads,
                dropout=0.0,
                batch_size=batch_data[0].shape[0],
                device=device,
            )  # type: ignore

            model_t = load_model(
                model_t, model_dir, trained_model_name=model_name, device=device
            )

            system_features = selected_features(system)

            if tm == "latest":

                output_dic, train_loss = infer_recon(model_t, train_loaders, device)

                for i in range(10):
                    tack_time = train_time[i * s_len : (i + 1) * s_len]
                    plot_name = "_".join(model_parameters[:8])
                    plot_output_recon(
                        output_dic['inputs'], output_dic['outputs'], i, system_features, tack_time, plot_name
                    )

                # Get reconstruction loss threshold.
                threshold = np.percentile(train_loss, 97)
                print("Reconstruction error threshold: ", threshold)

                start_time = timer()
                output_dic, test_loss = infer_recon(model_t, test_loaders, device)
                end_time = timer()

                print(
                    f"[INFO] Total inference time: {end_time - start_time:.3f} seconds"
                )

                # Detect all the samples which are anomalies.
                anomalies = test_loss > threshold
                anomalies = np.any(anomalies, axis=1)
                print("Number of anomaly samples: ", np.sum(anomalies))
                print("Indices of anomaly samples: ", np.where(anomalies)[0])
                anomalies_indices = np.where(anomalies)[0]

    	        ###### NOTE NOTE NOTE ######
                ###### NOTE NOTE NOTE ######
                # failures = pd.read_csv(args.RAW_DIR / "T06_failures_2017.csv")
                # matching_rows = failures[failures['Component'].str.lower().str.contains(system.lower(), case=False, na=False)]
                # failure_timestamps = matching_rows['Timestamp']

                # failure_timestamps = pd.to_datetime(failure_timestamps)
                # step_size = test_time.shape[0] // test_data_len # type: ignore
                # ind = np.arange(0, test_time.shape[0], step_size) # type: ignore
                # _test_time = test_time[ind] # type: ignore


                # indx = []
                # for failure_timestamp in failure_timestamps:
                #     # Find the closest time before the failure timestamp
                #     closest_time_before_failure = _test_time[test_time < failure_timestamp].max() # type: ignore

                #     # Find the index of the closest time
                #     index = np.where(_test_time == closest_time_before_failure)[0]
                #     if len(index) > 0:
                #         indx.append(index[0])


                # n = 5
                # for i in range(0, n*s_len, 16):
                #     temp = indx[0] - n*s_len + i
                #     time_stamp = test_time[indx[0] - n*s_len + i : indx[0] - (n-1)*s_len + i] # type: ignore
                #     plot_name = "_".join(model_parameters[:8])
                #     anomaly_plot(output_dic['inputs'], output_dic['outputs'], temp, system_features, time_stamp, plot_name)
                ###### NOTE NOTE NOTE ######
                ###### NOTE NOTE NOTE ######

                # output_dic["anomalies"] = anomalies
                # for i in anomalies_indices:
                #     time_stamp = test_time[i * s_len//2 : ((i + 1) * s_len//2) + s_len//2] # type: ignore
                #     plot_name = "_".join(model_parameters[:8])
                #     anomaly_plot(output_dic['inputs'], output_dic['outputs'], i, system_features, time_stamp, plot_name)

                # Save the results
                os.makedirs(f"{os.getcwd()}/results/anomalies/", exist_ok=True)
                np.savetxt(f"{os.getcwd()}/results/anomalies/{model_name[:-4]}_indices.csv", anomalies_indices, delimiter=",")

                features = selected_features(system)
                inputs = pd.DataFrame(output_dic["inputs"].reshape(-1, len(selected_features(system))), columns=features)
                outputs = pd.DataFrame(output_dic["outputs"].reshape(-1, len(selected_features(system))), columns=features)
                time_stamps = pd.Series(test_time[:inputs.shape[0]]) # type: ignore

                input_df = pd.concat((time_stamps, inputs), axis=1)
                columns = ["Timestamp", *features]
                input_df.columns = columns
                os.makedirs(f"{os.getcwd()}/results/preds/", exist_ok=True)
                input_df.to_csv(
                    f"{os.getcwd()}/results/preds/{model_name[:-4]}_{tm}_input.csv",
                    index=False,
                )

                recon_df = pd.concat((time_stamps, outputs), axis=1)
                recon_df.columns = columns
                recon_df.to_csv(
                    f"{os.getcwd()}/results/preds/{model_name[:-4]}_{tm}_recon.csv",
                    index=False,
                )
                break

        #     # Start the timer
        #     start_time = timer()

        #     y_true, y_pred = infer(
        #         model_t, train_loaders, loss_fn, [0, 0.5], model_name[:-4], device
        #     )
        #     # End the timer and print out how long it took
        #     end_time = timer()
        #     print(f"[INFO] Total inference time: {end_time - start_time:.3f} seconds")

        #     # compute average rmse and mae.
        #     # save the metric along with models parameters
        #     rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
        #     mae = np.mean(np.abs(y_true - y_pred))
        #     print(f"RMSE: {rmse}, MAE: {mae}")

        #     results["batch_size"].append(b_size)
        #     results["seq_len"].append(s_len)
        #     results["hidden_size"].append(h_size)
        #     results["layers"].append(n_layers)
        #     results["n_heads"].append(n_heads)
        #     results["l_rate"].append(l_rate)
        #     results["dropout"].append(dropout)
        #     results["RMSE"].append(rmse)
        #     results["MAE"].append(mae)

        # # save models_results file as csv
        # if tm != "latest":
        #     df = pd.DataFrame(data=results)
        #     _dir = f"{str(args.RAW_DIR)[:-4]}/results/"
        #     os.makedirs(_dir, exist_ok=True)
        #     df.to_csv(f"{_dir}{date_string}_results.csv", index=False)

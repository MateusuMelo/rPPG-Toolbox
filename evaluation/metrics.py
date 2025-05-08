import numpy as np
import pandas as pd
import torch
from evaluation.post_process import *
from tqdm import tqdm
from evaluation.BlandAltmanPy import BlandAltman
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.utils.validation import check_consistent_length

def read_label(dataset):
    """Read manually corrected labels."""
    df = pd.read_csv("label/{0}_Comparison.csv".format(dataset))
    out_dict = df.to_dict(orient='index')
    out_dict = {str(value['VideoID']): value for key, value in out_dict.items()}
    return out_dict


def read_hr_label(feed_dict, index):
    """Read manually corrected UBFC labels."""
    # For UBFC only
    if index[:7] == 'subject':
        index = index[7:]
    video_dict = feed_dict[index]
    if video_dict['Preferred'] == 'Peak Detection':
        hr = video_dict['Peak Detection']
    elif video_dict['Preferred'] == 'FFT':
        hr = video_dict['FFT']
    else:
        hr = video_dict['Peak Detection']
    return index, hr


def _reform_data_from_dict(data, flatten=True):
    """Helper func for calculate metrics: reformat predictions and labels from dicts. """
    sort_data = sorted(data.items(), key=lambda x: x[0])
    sort_data = [i[1] for i in sort_data]
    sort_data = torch.cat(sort_data, dim=0)

    if flatten:
        sort_data = np.reshape(sort_data.cpu(), (-1))
    else:
        sort_data = np.array(sort_data.cpu())

    return sort_data


def calculate_rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))


def calculate_standard_error(y_true, y_pred, num_samples):
    return np.std(np.abs(y_pred - y_true)) / np.sqrt(num_samples)


def calculate_mape(y_true, y_pred):
    check_consistent_length(y_true, y_pred)
    y_true, y_pred = np.array(y_true), np.array(y_pred)

    mask = y_true != 0
    y_true = y_true[mask]
    y_pred = y_pred[mask]
    if len(y_true) == 0:
        return np.nan
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100


def calculate_pearson(y_true, y_pred):
    return np.corrcoef(y_true, y_pred)

def bvp_bpm_extract(prediction, label, config):
    video_frame_size = prediction.shape[0]
    gt_hr_all = list()
    predict_hr_all = list()
    SNR_all = list()
    MACC_all = list()
    if "unsupervised" in config.TOOLBOX_MODE:
        FS = config.UNSUPERVISED.DATA.FS
    else:
        FS = config.TEST.DATA.FS

    if config.INFERENCE.EVALUATION_WINDOW.USE_SMALLER_WINDOW:
        window_frame_size = config.INFERENCE.EVALUATION_WINDOW.WINDOW_SIZE * FS
        if window_frame_size > video_frame_size:
            window_frame_size = video_frame_size
    else:
        window_frame_size = video_frame_size

    for i in range(0, len(prediction), window_frame_size):
        pred_window = prediction[i:i + window_frame_size]
        label_window = label[i:i + window_frame_size]

        if len(pred_window) < 9:
            print(
                f"Window frame size of {len(pred_window)} is smaller than minimum pad length of 9. Window ignored!")
            continue

        if (config.TEST.DATA.PREPROCESS.LABEL_TYPE == "Standardized" or
                config.TEST.DATA.PREPROCESS.LABEL_TYPE == "Raw"):
            diff_flag_test = False
        elif config.TEST.DATA.PREPROCESS.LABEL_TYPE == "DiffNormalized":
            diff_flag_test = True
        elif config.UNSUPERVISED.DATA.PREPROCESS.LABEL_TYPE == "Raw":
            diff_flag_test = False
        else:
            raise ValueError("Unsupported label type in testing!")

        if config.INFERENCE.EVALUATION_METHOD == "peak detection":
            gt_hr, pred_hr, SNR, macc = calculate_metric_per_video(
                pred_window, label_window, diff_flag=diff_flag_test, fs=FS, hr_method='Peak')
            gt_hr_all.append(gt_hr)
            predict_hr_all.append(pred_hr)
            SNR_all.append(SNR)
            MACC_all.append(macc)
        elif config.INFERENCE.EVALUATION_METHOD == "FFT":
            gt_hr, pred_hr, SNR, macc = calculate_metric_per_video(
                pred_window, label_window, diff_flag=diff_flag_test, fs=FS, hr_method='FFT')
            gt_hr_all.append(gt_hr)
            predict_hr_all.append(pred_hr)
            SNR_all.append(SNR)
            MACC_all.append(macc)
        else:
            raise ValueError("Inference evaluation method name wrong!")
    return gt_hr_all, predict_hr_all, SNR_all, MACC_all

def calculate_metrics(predictions, labels, config):
    """Calculate rPPG Metrics (MAE, RMSE, MAPE, Pearson Coef.)."""
    predict_hr_fft_all = list()
    gt_hr_fft_all = list()
    predict_hr_peak_all = list()
    gt_hr_peak_all = list()
    SNR_all = list()
    MACC_all = list()
    print("Calculating metrics!")
    if "unsupervised" in config.TOOLBOX_MODE:
        FS = config.UNSUPERVISED.DATA.FS
        metrics = config.UNSUPERVISED.METRICS
    else:
        FS = config.TEST.DATA.FS
        metrics = config.TEST.METRICS
    for index in tqdm(predictions.keys(), ncols=80):
        if "unsupervised" in config.TOOLBOX_MODE:
            prediction = predictions[index][0]
            label = labels[index][0]
        else:
            prediction = _reform_data_from_dict(predictions[index])
            label = _reform_data_from_dict(labels[index])
        video_frame_size = prediction.shape[0]
        if config.INFERENCE.EVALUATION_WINDOW.USE_SMALLER_WINDOW:
            window_frame_size = config.INFERENCE.EVALUATION_WINDOW.WINDOW_SIZE * FS
            if window_frame_size > video_frame_size:
                window_frame_size = video_frame_size
        else:
            window_frame_size = video_frame_size

        for i in range(0, len(prediction), window_frame_size):
            pred_window = prediction[i:i + window_frame_size]
            label_window = label[i:i + window_frame_size]

            if len(pred_window) < 9:
                print(
                    f"Window frame size of {len(pred_window)} is smaller than minimum pad length of 9. Window ignored!")
                continue

            if (config.TEST.DATA.PREPROCESS.LABEL_TYPE == "Standardized" or
                    config.TEST.DATA.PREPROCESS.LABEL_TYPE == "Raw"):
                diff_flag_test = False
            elif config.TEST.DATA.PREPROCESS.LABEL_TYPE == "DiffNormalized":
                diff_flag_test = True
            elif config.UNSUPERVISED.DATA.PREPROCESS.LABEL_TYPE == "Raw":
                diff_flag_test = False
            else:
                raise ValueError("Unsupported label type in testing!")

            if config.INFERENCE.EVALUATION_METHOD == "peak detection":
                gt_hr_peak, pred_hr_peak, SNR, macc = calculate_metric_per_video(
                    pred_window, label_window, diff_flag=diff_flag_test, fs=FS, hr_method='Peak')
                gt_hr_peak_all.append(gt_hr_peak)
                predict_hr_peak_all.append(pred_hr_peak)
                SNR_all.append(SNR)
                MACC_all.append(macc)
            elif config.INFERENCE.EVALUATION_METHOD == "FFT":
                gt_hr_fft, pred_hr_fft, SNR, macc = calculate_metric_per_video(
                    pred_window, label_window, diff_flag=diff_flag_test, fs=FS, hr_method='FFT')
                gt_hr_fft_all.append(gt_hr_fft)
                predict_hr_fft_all.append(pred_hr_fft)
                SNR_all.append(SNR)
                MACC_all.append(macc)
            else:
                raise ValueError("Inference evaluation method name wrong!")

    # Filename ID to be used in any results files (e.g., Bland-Altman plots) that get saved
    if config.TOOLBOX_MODE == 'train_and_test':
        filename_id = config.TRAIN.MODEL_FILE_NAME
    elif config.TOOLBOX_MODE == 'only_test':
        model_file_root = config.INFERENCE.MODEL_PATH.split("/")[-1].split(".pth")[0]
        filename_id = model_file_root + "_" + config.TEST.DATA.DATASET
    elif config.TOOLBOX_MODE == 'unsupervised_method':
        filename_id = str(predictions.keys()) + "_" + config.UNSUPERVISED.DATA.DATASET
    else:
        raise ValueError('Metrics.py evaluation only supports train_and_test and only_test!')

    if config.INFERENCE.EVALUATION_METHOD == "FFT":
        gt_hr_fft_all = np.array(gt_hr_fft_all)
        predict_hr_fft_all = np.array(predict_hr_fft_all)
        SNR_all = np.array(SNR_all)
        MACC_all = np.array(MACC_all)
        num_test_samples = len(predict_hr_fft_all)
        for metric in metrics:
            if metric == "MAE":
                MAE_FFT = mean_absolute_error(gt_hr_fft_all, predict_hr_fft_all)
                standard_error = calculate_standard_error(gt_hr_fft_all, predict_hr_fft_all, num_test_samples)
                print("FFT MAE (FFT Label): {0} +/- {1}".format(MAE_FFT, standard_error))
            elif metric == "RMSE":
                # Calculate the squared errors, then RMSE, in order to allow
                # for a more robust and intuitive standard error that won't
                # be influenced by abnormal distributions of errors.

                RMSE_FFT = calculate_rmse(gt_hr_fft_all, predict_hr_fft_all)
                standard_error = calculate_standard_error(gt_hr_fft_all, predict_hr_fft_all, num_test_samples)
                print("FFT RMSE (FFT Label): {0} +/- {1}".format(RMSE_FFT, standard_error))
            elif metric == "MAPE":
                MAPE_FFT = calculate_mape(gt_hr_fft_all, predict_hr_fft_all) * 100
                standard_error = calculate_standard_error(gt_hr_fft_all,predict_hr_fft_all,num_test_samples) * 100
                print("FFT MAPE (FFT Label): {0} +/- {1}".format(MAPE_FFT, standard_error))
            elif metric == "Pearson":
                Pearson_FFT = calculate_pearson(gt_hr_fft_all, predict_hr_fft_all)
                correlation_coefficient = Pearson_FFT[0][1]
                standard_error = np.sqrt((1 - correlation_coefficient ** 2) / (num_test_samples - 2))
                print("FFT Pearson (FFT Label): {0} +/- {1}".format(correlation_coefficient, standard_error))
            elif metric == "SNR":
                SNR_FFT = np.mean(SNR_all)
                standard_error = np.std(SNR_all) / np.sqrt(num_test_samples)
                print("FFT SNR (FFT Label): {0} +/- {1} (dB)".format(SNR_FFT, standard_error))
            elif metric == "MACC":
                MACC_avg = np.mean(MACC_all)
                standard_error = np.std(MACC_all) / np.sqrt(num_test_samples)
                print("FFT MACC (FFT Label): {0} +/- {1}".format(MACC_avg, standard_error))
            elif "AU" in metric:
                pass
            elif "BA" in metric:
                compare = BlandAltman(gt_hr_fft_all, predict_hr_fft_all, config, averaged=True)
                compare.scatter_plot(
                    x_label='GT PPG HR [bpm]',
                    y_label='rPPG HR [bpm]',
                    show_legend=True, figure_size=(5, 5),
                    the_title=f'{filename_id}_FFT_BlandAltman_ScatterPlot',
                    file_name=f'{filename_id}_FFT_BlandAltman_ScatterPlot.pdf')
                compare.difference_plot(
                    x_label='Difference between rPPG HR and GT PPG HR [bpm]',
                    y_label='Average of rPPG HR and GT PPG HR [bpm]',
                    show_legend=True, figure_size=(5, 5),
                    the_title=f'{filename_id}_FFT_BlandAltman_DifferencePlot',
                    file_name=f'{filename_id}_FFT_BlandAltman_DifferencePlot.pdf')
            else:
                raise ValueError("Wrong Test Metric Type")
    elif config.INFERENCE.EVALUATION_METHOD == "peak detection":
        gt_hr_peak_all = np.array(gt_hr_peak_all)
        predict_hr_peak_all = np.array(predict_hr_peak_all)
        SNR_all = np.array(SNR_all)
        MACC_all = np.array(MACC_all)
        num_test_samples = len(predict_hr_peak_all)
        for metric in metrics:
            if metric == "MAE":
                MAE_PEAK = mean_absolute_error(gt_hr_peak_all, predict_hr_peak_all)
                standard_error = calculate_standard_error(gt_hr_peak_all, predict_hr_peak_all, num_test_samples)
                print("Peak MAE (Peak Label): {0} +/- {1}".format(MAE_PEAK, standard_error))
            elif metric == "RMSE":
                # Calculate the squared errors, then RMSE, in order to allow
                # for a more robust and intuitive standard error that won't
                # be influenced by abnormal distributions of errors.

                RMSE_PEAK = calculate_rmse(gt_hr_peak_all, predict_hr_peak_all)
                standard_error = calculate_standard_error(gt_hr_peak_all, predict_hr_peak_all, num_test_samples)
                print("PEAK RMSE (Peak Label): {0} +/- {1}".format(RMSE_PEAK, standard_error))
            elif metric == "MAPE":
                MAPE_PEAK = calculate_mape(gt_hr_peak_all, predict_hr_peak_all)
                standard_error = calculate_standard_error(gt_hr_peak_all, predict_hr_peak_all, num_test_samples) * 100
                print("PEAK MAPE (Peak Label): {0} +/- {1}".format(MAPE_PEAK, standard_error))
            elif metric == "Pearson":
                Pearson_PEAK = calculate_pearson(gt_hr_peak_all, predict_hr_peak_all)
                correlation_coefficient = Pearson_PEAK[0][1]
                standard_error = np.sqrt((1 - correlation_coefficient ** 2) / (num_test_samples - 2))
                print("PEAK Pearson (Peak Label): {0} +/- {1}".format(correlation_coefficient, standard_error))
            elif metric == "SNR":
                SNR_PEAK = np.mean(SNR_all)
                standard_error = np.std(SNR_all) / np.sqrt(num_test_samples)
                print("PEAK SNR (PEAK Label): {0} +/- {1} (dB)".format(SNR_PEAK, standard_error))
            elif metric == "MACC":
                MACC_avg = np.mean(MACC_all)
                standard_error = np.std(MACC_all) / np.sqrt(num_test_samples)
                print("PEAK MACC (PEAK Label): {0} +/- {1}".format(MACC_avg, standard_error))
            elif "AU" in metric:
                pass
            elif "BA" in metric:
                compare = BlandAltman(gt_hr_peak_all, predict_hr_peak_all, config, averaged=True)
                compare.scatter_plot(
                    x_label='GT PPG HR [bpm]',
                    y_label='rPPG HR [bpm]',
                    show_legend=True, figure_size=(5, 5),
                    the_title=f'{filename_id}_Peak_BlandAltman_ScatterPlot',
                    file_name=f'{filename_id}_Peak_BlandAltman_ScatterPlot.pdf')
                compare.difference_plot(
                    x_label='Difference between rPPG HR and GT PPG HR [bpm]',
                    y_label='Average of rPPG HR and GT PPG HR [bpm]',
                    show_legend=True, figure_size=(5, 5),
                    the_title=f'{filename_id}_Peak_BlandAltman_DifferencePlot',
                    file_name=f'{filename_id}_Peak_BlandAltman_DifferencePlot.pdf')
            else:
                raise ValueError("Wrong Test Metric Type")
    else:
        raise ValueError("Inference evaluation method name wrong!")

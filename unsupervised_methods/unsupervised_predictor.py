"""Unsupervised learning methods including POS, GREEN, CHROME, ICA, LGI and PBV."""
import numpy as np
import pandas as pd
from unsupervised_methods.methods.CHROME_DEHAAN import *
from unsupervised_methods.methods.GREEN import *
from unsupervised_methods.methods.ICA_POH import *
from unsupervised_methods.methods.LGI import *
from unsupervised_methods.methods.PBV import *
from unsupervised_methods.methods.POS_WANG import *
from unsupervised_methods.methods.OMIT import *
from tqdm import tqdm
from evaluation.metrics import calculate_metrics, calculate_mape, calculate_mae, calculate_rmse, bvp_bpm_extract

def unsupervised_predict(config, data_loader):
    """ Model evaluation on the testing dataset."""
    if data_loader["unsupervised"] is None:
        raise ValueError("No data for unsupervised method predicting")

    for method_name in config.UNSUPERVISED.METHOD:
        preds_dict_bvp = dict()
        labels_dict_bvp = dict()
        if method_name not in preds_dict_bvp.keys():
            preds_dict_bvp[method_name] = dict()
            labels_dict_bvp[method_name] = dict()
        print("===Unsupervised Method ( " + method_name + " ) Predicting ===")
        sbar = tqdm(data_loader["unsupervised"], ncols=80)
        for _, test_batch in enumerate(sbar):
            batch_size = test_batch[0].shape[0]
            for idx in range(batch_size):
                data_input, labels_input = test_batch[0][idx].cpu().numpy(), test_batch[1][idx].cpu().numpy()
                data_input = data_input[..., :3]
                sort_index = int(test_batch[3][idx])
                if method_name == "POS":
                    BVP = POS_WANG(data_input, config.UNSUPERVISED.DATA.FS)
                elif method_name == "CHROM":
                    BVP = CHROME_DEHAAN(data_input, config.UNSUPERVISED.DATA.FS)
                elif method_name == "ICA":
                    BVP = ICA_POH(data_input, config.UNSUPERVISED.DATA.FS)
                elif method_name == "GREEN":
                    BVP = GREEN(data_input)
                elif method_name == "LGI":
                    BVP = LGI(data_input)
                elif method_name == "PBV":
                    BVP = PBV(data_input)
                elif method_name == "OMIT":
                    BVP = OMIT(data_input)
                else:
                    raise ValueError("unsupervised method name wrong!")
                preds_dict_bvp[method_name][sort_index] = np.asarray(BVP)
                labels_dict_bvp[method_name][sort_index] = np.asarray(labels_input)
        calculate_metrics(preds_dict_bvp,labels_dict_bvp,config)

def unsupervised_roi_predict(config, data_loader):
    """ Model evaluation on the testing dataset."""
    if data_loader["unsupervised"] is None:
        raise ValueError("No data for unsupervised method predicting")

    # Inicializa os dicionários principais
    preds_dict_bvp = {method: {} for method in config.UNSUPERVISED.METHOD}
    results = pd.DataFrame()
    for method_name in config.UNSUPERVISED.METHOD:
        print("===Unsupervised Method ( " + method_name + " ) Predicting ===")

        gt_hr_all = []
        predict_hr_all = []
        sbar = tqdm(data_loader["unsupervised"], ncols=80)
        for _, test_batch in enumerate(sbar):
            batch_size = test_batch[0].shape[0]
            for idx in range(batch_size):
                data_input, labels_input = test_batch[0][idx].cpu().numpy(), test_batch[1][idx].cpu().numpy()
                data_input = data_input[..., :3]

                # Processa com o método selecionado
                if method_name == "POS":
                    BVP = POS_WANG(data_input, config.UNSUPERVISED.DATA.FS)
                elif method_name == "CHROM":
                    BVP = CHROME_DEHAAN(data_input, config.UNSUPERVISED.DATA.FS)
                elif method_name == "ICA":
                    BVP = ICA_POH(data_input, config.UNSUPERVISED.DATA.FS)
                elif method_name == "GREEN":
                    BVP = GREEN(data_input)
                elif method_name == "LGI":
                    BVP = LGI(data_input)
                elif method_name == "PBV":
                    BVP = PBV(data_input)
                elif method_name == "OMIT":
                    BVP = OMIT(data_input)
                else:
                    raise ValueError("unsupervised method name wrong!")

                gt_hr , predict_hr, _, _ = bvp_bpm_extract(BVP,labels_input,config)
                gt_hr_all.extend(gt_hr)
                predict_hr_all.extend(predict_hr)

            # Obtém o número da ROI
            _, roi = ((test_batch[2][0]).split('-roi'))
            roi = int(roi)
            preds_dict_bvp[method_name][roi] = [np.asarray(gt_hr_all),np.asarray(predict_hr_all)]

    for method_name in config.UNSUPERVISED.METHOD:
        for roi in preds_dict_bvp[method_name]:
            y_true, y_pred = preds_dict_bvp[method_name][roi]
            mae = calculate_mae(y_true, y_pred)
            mape = calculate_mape(y_true, y_pred)
            mse = calculate_rmse(y_true, y_pred)
            results = results.append({'ROI': roi, 'Method': method_name, 'MAE': mae, 'MAPE': mape, 'RMSE': mse},
                                     ignore_index=True)

    best_results = results.loc[results.groupby('Method')['MAE'].idxmin()].reset_index(drop=True)
    print("\n Top 5 best results:\n")
    print(f"{'Method':<10} | {'ROI':<5} | {'MAE':<8} | {'MAPE':<8} | {'RMSE':<8}")
    print("-" * 50)
    for _, row in best_results.sort_values(by='MAE').iterrows():
        print(
            f"{row['Method']:<10} | {int(row['ROI']):<5} | {row['MAE']:<8.4f} | {row['MAPE']:<8.4f} | {row['RMSE']:<8.4f}")
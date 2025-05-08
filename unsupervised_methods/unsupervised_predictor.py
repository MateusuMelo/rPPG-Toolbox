"""Unsupervised learning methods including POS, GREEN, CHROME, ICA, LGI and PBV."""
import pandas as pd
from unsupervised_methods.methods.CHROME_DEHAAN import *
from unsupervised_methods.methods.GREEN import *
from unsupervised_methods.methods.ICA_POH import *
from unsupervised_methods.methods.LGI import *
from unsupervised_methods.methods.PBV import *
from unsupervised_methods.methods.POS_WANG import *
from unsupervised_methods.methods.OMIT import *
from tqdm import tqdm
from evaluation.metrics import calculate_metrics, calculate_mape, bvp_bpm_extract
from sklearn.metrics import mean_squared_error, mean_absolute_error

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
    """Model evaluation on the testing dataset."""
    if data_loader["unsupervised"] is None:
        raise ValueError("No data for unsupervised method predicting")

    preds_dict_bvp = {method: {} for method in config.UNSUPERVISED.METHOD}
    results = pd.DataFrame()

    for method_name in config.UNSUPERVISED.METHOD:
        print(f"=== Processing Method: {method_name} ===")
        method_results = {roi: {'gt': [], 'pred': []} for roi in range(1, 21)}

        sbar = tqdm(data_loader["unsupervised"], ncols=80)
        for _, test_batch in enumerate(sbar):
            _, roi_str = test_batch[2][0].split('-roi')
            roi = int(roi_str)

            batch_size = test_batch[0].shape[0]
            for idx in range(batch_size):
                data_input, labels_input = test_batch[0][idx].cpu().numpy(), test_batch[1][idx].cpu().numpy()
                data_input = data_input[..., :3]

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
                    raise ValueError("Unsupervised method name not recognized!")

                gt_hr, predict_hr, _, _ = bvp_bpm_extract(BVP, labels_input, config)

                method_results[roi]['gt'].extend(gt_hr)
                method_results[roi]['pred'].extend(predict_hr)

        for roi in method_results:
            if method_results[roi]['gt']:
                preds_dict_bvp[method_name][roi] = (
                    np.array(method_results[roi]['gt']),
                    np.array(method_results[roi]['pred'])
                )

    for method_name in preds_dict_bvp:
        for roi in preds_dict_bvp[method_name]:
            y_true, y_pred = preds_dict_bvp[method_name][roi]

            mae = mean_absolute_error(y_true, y_pred)
            mape = calculate_mape(y_true, y_pred)
            rmse = np.sqrt(mean_squared_error(y_true, y_pred))

            results = results.append({
                'Method': method_name,
                'ROI': roi,
                'MAE': mae,
                'MAPE': mape,
                'RMSE': rmse,
                'Samples': len(y_true)
            }, ignore_index=True)
    best_results = results.loc[results.groupby('Method')['MAE'].idxmin()].reset_index(drop=True)

    print("\nTop Results per Method:")
    print(f"{'Method':<10} | {'ROI':<5} | {'MAE':<8} | {'MAPE':<8} | {'RMSE':<8} | {'Samples':<8}")
    print("-" * 65)
    for _, row in best_results.sort_values(by='MAE').iterrows():
        print(f"{row['Method']:<10} | {int(row['ROI']):<5} | {row['MAE']:<8.4f} | "
              f"{row['MAPE']:<8.4f} | {row['RMSE']:<8.4f} | {row['Samples']:<8}")

    return results
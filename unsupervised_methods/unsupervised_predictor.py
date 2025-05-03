"""Unsupervised learning methods including POS, GREEN, CHROME, ICA, LGI and PBV."""
from unsupervised_methods.methods.CHROME_DEHAAN import *
from unsupervised_methods.methods.GREEN import *
from unsupervised_methods.methods.ICA_POH import *
from unsupervised_methods.methods.LGI import *
from unsupervised_methods.methods.PBV import *
from unsupervised_methods.methods.POS_WANG import *
from unsupervised_methods.methods.OMIT import *
from tqdm import tqdm
from evaluation.metrics import calculate_metrics

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
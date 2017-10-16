#!/usr/bin/env python

from fintech_analysis import WeightVeiwer
from fintech_analysis.models import FeatureExtractRegressionModel

if __name__ == '__main__':
    model = FeatureExtractRegressionModel(batch_input_shape=(None, 1,37, 1),
                                          out_dim=1, kernel_size=(1, 1))
    model.load_weights("/Users/Eddie/fintech_tutorial/reg_w0.h5")
    model.summary()
    w = WeightVeiwer(model)
    w.set_col_names(['1', '2', '3', '4', '5',
                     '6', '7', '8','9','10','11','12', '13', '14', '15','16','17','18','19','20','21','22','23','24','25','26','27','28','29','30','31','32','33','34','35','36','37'])
    w.show_pn_bar()

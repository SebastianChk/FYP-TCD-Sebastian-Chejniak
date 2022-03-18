import pandas as pd
from functions import CreateItemEncoder,  AddAgeFeature, CreateDateScalers, loadCal, CreateDateFeatures, EncodeItems, ConvertDTypes

#############################################################
# Note: Many of these funtions automatically pickle objects #
# to a specific path. More details can be found in          #
# functions.py                                              #
#############################################################

ste = pd.read_csv('../../data/external/sales_train_evaluation.csv')
# Creates sklearn encoder object and pickles it
ste.pipe(CreateItemEncoder)

cal = loadCal()
# Creates sklearn scaler to convert date features to standardised features, and pickles scaler
cal.pipe(
    AddAgeFeature).pipe(
    CreateDateScalers)


ste = pd.read_csv('../../data/external/sales_train_evaluation.csv')

ste_processed = ste \
    .pipe(EncodeItems) \
    .pipe(ConvertDTypes) \
    .set_index('id')

ste_processed.to_pickle('../../data/processed/ste_processed.pkl')

cal = loadCal()

cal_processed = cal \
    .pipe(AddAgeFeature) \
    .pipe(CreateDateFeatures)

cal_processed.to_pickle('../../data/processed/cal_processed.pkl')

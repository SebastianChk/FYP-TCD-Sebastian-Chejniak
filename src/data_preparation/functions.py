import pandas as pd
import numpy as np
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from gluonts.dataset.common import ListDataset
from gluonts.dataset.field_names import FieldName
import pickle

def StartPipeline(df):
    return df.copy()


def CreateItemEncoder(ste, file_location='../../data/interim/', file_name= 'item_encoder.pkl', exclude=[]):
    catCols = ['id', 'item_id', 'dept_id', 'cat_id', 'store_id', 'state_id']
    catCols = [catCol for catCol in catCols if catCol not in exclude]
    cat_encoder = OrdinalEncoder()
    cat_encoder.fit(ste[catCols])
    
    with open(file_location + file_name, 'wb') as f:
        pickle.dump(cat_encoder, f)
    
    
def SortByID(ste, id_list_file='../../data/interim'):
    """Sort dataframe rows by id column, using id_list_file"""
    with open(id_list_file, 'r') as f:
        id_list = json.loads(f.read())
    
    def sorter(column):
        """Sort function"""
        correspondence = {ID: order for order, ID in enumerate(id_list)}
        return column.map(correspondence)
    
    return ste.sort_values(by='id', key=sorter)
            
    
def EncodeItems(ste, file_location='../../data/interim/', file_name='item_encoder.pkl', exclude=[]):
    with open(file_location + file_name, 'rb') as f:
        cat_encoder = pickle.load(f)
    catCols = ['id', 'item_id', 'dept_id', 'cat_id', 'store_id', 'state_id']
    newcatCols = [catCol + '_encoded' for catCol in catCols if catCol not in exclude]
    ste[newcatCols] = ste[catCols].pipe(cat_encoder.transform)
    return ste
    
    
def ConvertDTypes(ste):
    """ste must have encoded categorical columns with the following names"""
    # may be redundant in some cases
    catCols = ['id', 'item_id', 'dept_id', 'cat_id', 'store_id', 'state_id']
    catCols = [i + '_encoded' for i in catCols]
    ndays = 1941
    numCols = [f'd_{daynum}' for daynum in range(1, ndays+1)]
    
    # dtypes is a dictionary mapping each column to the desired datatype
    dtypes = {numCol: "uint16" for numCol in numCols}
    #catdtypes = {catCol:('uint8' if catCol != 'item_id' else 'uint16') for catCol in catCols[1:]}
    catdtypes = {catCol:('uint8' if catCol != 'item_id_encoded' and catCol != 'id_encoded' else 'uint16') for catCol in catCols}
    dtypes.update(catdtypes)
    
    return ste.astype(dtypes)


def loadCal(last_date='2016-05-22'):
    """Automatically excludes not needed last 28 days,
    reads date column as a datetime object,
    and sets date as index"""
    cal = pd.read_csv('../../data/external/calendar.csv',
                      parse_dates=['date'],
                      index_col=0)
    return cal.loc[:last_date,:]


def AddAgeFeature(cal):
    """Returns cal with a new column 'age', which is a linearly increasing age feature """
    cal['age'] = np.arange(0, len(cal))
    return cal


def CreateDateScalers(cal):
    """Creates and pickles a sklearn scaler which standardises the four date feature columns (for deepAR only as of 29-12-2021)"""
    dateScaler = StandardScaler()
    dateCols = ['wday', 'month', 'year', 'age']
    
    with open(f'../../data/interim/dateScaler.pkl', 'wb') as f:
        pickle.dump(dateScaler.fit(cal[dateCols]) , f)
        
        
def CreateDateFeatures(cal):

    with open(f'../../data/interim/dateScaler.pkl', 'rb') as f:
        dateScaler = pickle.load(f)
    
    dateCols = ['wday', 'month', 'year', 'age']
    normDateCols = [i + '_norm' for i in dateCols]
    
    cal[normDateCols] = dateScaler.transform(cal[dateCols])
    return cal


def CreateListDatasets(ste, cal,
                      prediction_length = 28,
                      freq = "D",
                      start = pd.Timestamp("29-01-2011"),
                      exclude=[], 
                      prediction=False):

    catCols = ['id', 'item_id', 'dept_id', 'cat_id', 'store_id', 'state_id']
    catCols = [catCol + '_encoded' for catCol in catCols if catCol not in exclude]
    
    ndays = 1941
    numCols = [f'd_{daynum}' for daynum in range(1, ndays+1)]
    
    normDateCols = ['wday', 'month', 'year', 'age']
    normDateCols = [i + '_norm' for i in normDateCols]
    
    timeseries = ste.loc[:, numCols].to_numpy()
    static = ste.loc[:, catCols].to_numpy()
    dynamic = cal[normDateCols].transpose().to_numpy()
    if not prediction:
        train = ListDataset([
                {        
                FieldName.TARGET: target,
                FieldName.START: start,
                FieldName.FEAT_STATIC_CAT: fsc,
                FieldName.FEAT_DYNAMIC_REAL: dynamic[:,:-prediction_length]
                } for (target, fsc) in zip(timeseries[:,:-prediction_length], static)
            ], freq=freq)

        test = ListDataset([
                {        
                FieldName.TARGET: target,
                FieldName.START: start,
                FieldName.FEAT_STATIC_CAT: fsc,
                FieldName.FEAT_DYNAMIC_REAL: dynamic
                } for (target, fsc) in zip(timeseries, static)
            ], freq=freq)

        # no. of unique categories for each static covariate (needed as input to deepAR estimator)
        cardinality = [ste.loc[:, col].nunique() for col in catCols]

        return train, test, cardinality
    
    train = ListDataset([
            {        
            FieldName.TARGET: target,
            FieldName.START: start,
            FieldName.FEAT_STATIC_CAT: fsc,
            FieldName.FEAT_DYNAMIC_REAL: dynamic[:,:-prediction_length]
            } for (target, fsc) in zip(timeseries[:,:-prediction_length], static)
        ], freq=freq)

    test = ListDataset([
            {        
            FieldName.TARGET: target,
            FieldName.START: start,
            FieldName.FEAT_STATIC_CAT: fsc,
            FieldName.FEAT_DYNAMIC_REAL: dynamic
            } for (target, fsc) in zip(timeseries, static)
        ], freq=freq)
    
    # no. of unique categories for each static covariate (needed as input to deepAR estimator)
    cardinality = [ste.loc[:, col].nunique() for col in catCols]
    
    return train, test, cardinality
    
    
    
# build dataset according to given 'dataset_file'
def build_dataset(args):
    dataset_name = getattr(args, 'dataset_file', '').upper()
    valid_datasets = {'SHHA', 'SHANGHAITECH', 'JHU', 'JHU_CROWD', 'JHU-CROWD++', 'MALL', 'MALL_DATASET', 'DEFAULT'}
    
    if dataset_name in valid_datasets or not dataset_name:
        from dataset.SHHA.loading_data import loading_data
        return loading_data

    # Fallback to standard loader
    from dataset.SHHA.loading_data import loading_data
    return loading_data

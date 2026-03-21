def flatten_dict(d, parent_key='', sep='.'):
    """
    Flattens a nested dictionary.
    Safe stringification is applied to values to ensure they do not exceed 
    MLflow's 250-character parameter limit.
    """
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            val_str = str(v)
            # MLflow parameters have a 250 character limit. Safely truncate if needed.
            if len(val_str) > 250:
                val_str = val_str[:247] + "..."
            items.append((new_key, val_str))
    return dict(items)

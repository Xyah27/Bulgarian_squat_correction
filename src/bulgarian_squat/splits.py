from sklearn.model_selection import train_test_split

def split_by_video(rep_meta, video_col='video_id', splits=None, seed=42):
    if splits is None:
        splits = {"train":0.7, "val":0.15, "test":0.15}
    vids = list({ m[video_col] for m in rep_meta })
    vids_train, vids_tmp = train_test_split(vids, test_size=(splits["val"]+splits["test"]), random_state=seed)
    vids_val, vids_test  = train_test_split(vids_tmp, test_size=splits["test"]/(splits["val"]+splits["test"]), random_state=seed)
    return set(vids_train), set(vids_val), set(vids_test)

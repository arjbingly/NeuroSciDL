from hashlib import sha256

def hash_df(df):
    return sha256(df.to_csv(index=False).encode()).hexdigest()

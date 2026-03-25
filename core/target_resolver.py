def resolve_target(df, user_target=None):
    if user_target:
        if user_target not in df.columns:
            raise ValueError(f"{user_target} not found in dataset")
        return user_target

    possible_targets = ["target", "label", "y", "class", "outcome"]

    for col in possible_targets:
        if col in df.columns:
            return col

    return df.columns[-1]
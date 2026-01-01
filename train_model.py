# scripts/train_model.py
import sys
import zipfile
import io
from pathlib import Path
import numpy as np
import joblib
import pickle

# require scikit-learn for the intended model
try:
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score
except ModuleNotFoundError as e:
    raise ModuleNotFoundError(
        "scikit-learn is required. Install with: pip install scikit-learn"
    ) from e

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def locate_signals():
    """Return ('dir', Path) or ('zip', Path) where signal files can be found."""
    dir_path = PROJECT_ROOT / "data" / "UCI HAR Dataset" / "train" / "Inertial Signals"
    zip_path = PROJECT_ROOT / "data" / "UCI_HAR_Dataset.zip"
    if dir_path.exists():
        return ("dir", dir_path)
    if zip_path.exists():
        return ("zip", zip_path)
    return (None, None)


def load_signal_from_store(filename: str):
    typ, src = locate_signals()
    if typ == "dir":
        path = src / filename
        if not path.exists():
            raise FileNotFoundError(f"{path} not found in extracted dataset.")
        return np.loadtxt(path)
    if typ == "zip":
        arcname = f"UCI HAR Dataset/train/Inertial Signals/{filename}"
        with zipfile.ZipFile(src) as z:
            if arcname not in z.namelist():
                raise FileNotFoundError(f"'{arcname}' not found inside {src}")
            with z.open(arcname) as f:
                return np.loadtxt(io.TextIOWrapper(f, encoding="utf-8"))
    raise FileNotFoundError(
        "Could not find dataset. Put extracted files under 'data/UCI HAR Dataset/' "
        "or the zip at 'data/UCI_HAR_Dataset.zip'."
    )


def extract_features(window_size=128, step=64):
    """Compute sliding-window mean/std and save 'features.npy' at project root."""
    print("Extracting features (window_size=%d, step=%d)..." % (window_size, step))
    acc_x = load_signal_from_store("body_acc_x_train.txt")
    acc_x = np.asarray(acc_x).ravel()
    features = []
    for i in range(0, acc_x.shape[0] - window_size + 1, step):
        w = acc_x[i : i + window_size]
        features.append([float(np.mean(w)), float(np.std(w))])
    features = np.array(features, dtype=float)
    out_path = PROJECT_ROOT / "features.npy"
    np.save(out_path, features)
    print(f"Saved features to {out_path} (shape: {features.shape})")
    return out_path


def find_or_create_features():
    """Find features.npy or create it by extracting from dataset."""
    candidates = [
        PROJECT_ROOT / "features.npy",
        Path.cwd() / "features.npy",
        Path(__file__).resolve().parent / "features.npy",
    ]
    for p in candidates:
        if p.exists():
            return p
    # Not found -> try to extract
    try:
        return extract_features()
    except Exception as e:
        raise FileNotFoundError(
            "features.npy not found and extraction failed: " + str(e)
        ) from e


def load_labels(filename="y_train.txt"):
    extracted = PROJECT_ROOT / "data" / "UCI HAR Dataset" / "train" / filename
    if extracted.exists():
        return np.loadtxt(extracted).astype(int)
    zip_p = PROJECT_ROOT / "data" / "UCI_HAR_Dataset.zip"
    if zip_p.exists():
        arc = f"UCI HAR Dataset/train/{filename}"
        with zipfile.ZipFile(zip_p) as z:
            if arc not in z.namelist():
                raise FileNotFoundError(f"'{arc}' not found in zip '{zip_p}'")
            with z.open(arc) as f:
                return np.loadtxt(io.TextIOWrapper(f, encoding="utf-8")).astype(int)
    raise FileNotFoundError(
        "Could not find labels; place extracted data in 'data/' or put the zip at 'data/UCI_HAR_Dataset.zip'."
    )


def map_labels(y_raw):
    y = []
    for l in y_raw:
        if l in (1, 2, 3):
            y.append(1)  # Walking
        elif l in (4, 5):
            y.append(0)  # Standing
        else:
            y.append(2)  # Abnormal
    return np.array(y, dtype=int)


def main():
    try:
        fx = find_or_create_features()
        X = np.load(fx)
        if X.ndim == 1:
            X = X.reshape(-1, 1)

        y_raw = load_labels()
        # if lengths mismatch, trim the longer side with a warning
        if len(y_raw) < len(X):
            print(
                f"Warning: fewer labels ({len(y_raw)}) than feature rows ({len(X)}). Trimming features to match labels."
            )
            X = X[: len(y_raw)]
        elif len(y_raw) > len(X):
            print(
                f"Warning: more labels ({len(y_raw)}) than feature rows ({len(X)}). Trimming labels to match features."
            )
            y_raw = y_raw[: len(X)]

        y = map_labels(y_raw)

        # Validate
        if np.isnan(X).any():
            raise ValueError("Features contain NaNs; please clean or re-extract.")
        if len(np.unique(y)) < 2:
            raise ValueError("Need at least two classes to train a classifier.")

        # Train/test split (using stratify)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        model = DecisionTreeClassifier(max_depth=4, min_samples_leaf=10, random_state=42)
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        print("Model accuracy:", acc)

        # Save model
        try:
            joblib.dump(model, "gait_model.pkl")
        except Exception:
            with open("gait_model.pkl", "wb") as f:
                pickle.dump(model, f)
        print("Model saved as gait_model.pkl")

    except Exception as e:
        print("ERROR:", e)
        sys.exit(1)


if __name__ == "__main__":
    main()

import io, os, sys, zipfile
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "data"); os.makedirs(OUT, exist_ok=True)


def aqs(raw, folder="aqs", fname="hourly_88101_2023.zip", tag="aqs", quantity="PM2.5", unit="ug/m3", name_="EPA-AQS-PM25-2023"):
    import pandas as pd
    path = os.path.join(raw, folder, fname)
    with zipfile.ZipFile(path) as z:
        name = [n for n in z.namelist() if n.endswith(".csv")][0]
        with z.open(name) as f:
            df = pd.read_csv(f, usecols=["State Code", "County Code", "Site Num", "POC", "Latitude", "Longitude",
                                         "Date GMT", "Time GMT", "Sample Measurement"],
                             dtype={"State Code": str, "County Code": str, "Site Num": str})
    df["site"] = df["State Code"] + "-" + df["County Code"] + "-" + df["Site Num"] + "-" + df["POC"].astype(str)
    df["t"] = pd.to_datetime(df["Date GMT"] + " " + df["Time GMT"])
    
    df["loc"] = df["State Code"] + "-" + df["County Code"] + "-" + df["Site Num"]
    first = df.groupby("loc")["POC"].transform("min")
    df = df[df["POC"] == first]
    wide = df.pivot_table(index="t", columns="loc", values="Sample Measurement", aggfunc="mean")
    idx = pd.date_range("2023-01-01 00:00", "2023-12-31 23:00", freq="h")
    wide = wide.reindex(idx)
    cover = wide.notna().mean()
    keep = cover[cover >= 0.90].index
    wide = wide[keep].ffill(limit=24).bfill().ffill()
    X = wide.to_numpy(float)
    assert not np.isnan(X).any()
    X = np.maximum(X, 0.0)
    coords = df.groupby("loc")[["Latitude", "Longitude"]].first().reindex(keep).to_numpy(float)
    np.savez_compressed(os.path.join(OUT, f"{tag}_field.npz"), X=X.astype(np.float32), latlon=coords, ids=np.array(keep),
                        unit=unit, quantity=quantity, name=name_, dt_s=3600.0, n_all=int(cover.size))
    print(tag, X.shape, "monitors in file", int(cover.size), "kept", len(keep), "mean", X.mean(), "sd", X.std())


def o3(raw):
    aqs(raw, folder="o3", fname="hourly_44201_2023.zip", tag="o3", quantity="ozone", unit="ppm", name_="EPA-AQS-O3-2023")


def pems(raw):
    path = os.path.join(raw, "pems", "pems_sf.zip")
    with zipfile.ZipFile(path) as z:
        names = z.namelist()
        print(names)

        def rd(fn):
            n = [m for m in names if m.endswith(fn)][0]
            return z.read(n).decode("ascii")
        perm = np.array(rd("randperm").strip().strip("[]").split(), int)
        days = []
        for part in ("PEMS_train", "PEMS_test"):
            for line in rd(part).strip().split("\n"):
                rows = line.strip().strip("[]").split(";")
                days.append(np.array([[float(v) for v in r.split()] for r in rows]))
    days = np.array(days)                 # [days, sensors, 144]
    print("days", days.shape, "perm", perm.shape)
    order = np.argsort(perm)
    days = days[order]
    X = np.concatenate([d.T for d in days[:60]], axis=0)
    np.savez_compressed(os.path.join(OUT, "pems_field.npz"), X=X.astype(np.float32), unit="occupancy", quantity="occupancy rate",
                        name="PEMS-SF", dt_s=600.0)
    print("pems", X.shape, "mean", X.mean(), "sd", X.std())


if __name__ == "__main__":
    raw = sys.argv[1]
    for what in (sys.argv[2:] or ["aqs", "o3", "pems"]):
        {"aqs": aqs, "o3": o3, "pems": pems}[what](raw)

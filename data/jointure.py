from pathlib import Path
import pandas as pd

# =========================
# Dossiers et années
# =========================
project_dir = Path(__file__).resolve().parent
data_dir = project_dir / "raw"
annees = range(2019, 2025)

# =========================
# LECTURE CSV
# =========================
def lire_csv(fichier):
    return pd.read_csv(fichier, sep=';', encoding='utf-8', low_memory=False)

# =========================
# CHARGEMENT DES TABLES
# =========================
def charger_table(table: str) -> pd.DataFrame:
    frames = []

    for annee in annees:
        fichiers = list(data_dir.glob(f"{table}*{annee}*.csv"))

        if not fichiers:
            print(f"⚠️ {table} {annee} manquant")
            continue

        fichier = fichiers[0]
        print(f"📁 {table} {annee}: {fichier.name}")

        data = lire_csv(fichier)
        data["annee"] = annee
        frames.append(data)

    if not frames:
        raise FileNotFoundError(f"Aucune table {table} trouvée")

    # harmonisation colonnes
    all_cols = {c for data in frames for c in data.columns}
    result = pd.concat(
        [data.reindex(columns=sorted(all_cols)) for data in frames],
        ignore_index=True
    )

    return result

# =========================
# NORMALISATION DES CLES ET NETTOYAGE
# =========================
def normaliser(data: pd.DataFrame) -> pd.DataFrame:
    # nettoyage colonnes
    data.columns = (
        data.columns.str.strip()
                  .str.lower()
                  .str.replace(" ", "_")
                  .str.replace("-", "_")
                  .str.replace("é", "e")
    )

    # nettoyage des colonnes texte
    for col in data.select_dtypes(include='object').columns:
        data[col] = data[col].str.replace('\xa0', '', regex=False).str.strip()

    # normalisation num_acc
    if "num_acc" in data.columns:
        data["num_acc"] = (
            data["num_acc"]
            .astype(str)
            .str.strip()
            .str.replace(".0", "", regex=False)
        )

    # normalisation id_vehicule
    if "id_vehicule" not in data.columns:
        for col in data.columns:
            if "vehicule" in col and ("id" in col or "lettre" in col):
                data.rename(columns={col: "id_vehicule"}, inplace=True)
                break

    if "id_vehicule" in data.columns:
        data["id_vehicule"] = (
            data["id_vehicule"]
            .astype(str)
            .str.strip()
            .str.replace(".0", "", regex=False)
        )

    # annees en int
    data["annee"] = data["annee"].astype(int)

    return data

# =========================
# MAIN
# =========================
def main():
    print("=== BAAC 2019–2024 - Jointure complète ===\n")

    print("1. Chargement des tables")
    caracteristiques = charger_table("caracteristiques")
    lieux = charger_table("lieux")
    vehicules = charger_table("vehicules")
    usagers = charger_table("usagers")

    # normalisation
    caracteristiques = normaliser(caracteristiques)
    lieux = normaliser(lieux)
    vehicules = normaliser(vehicules)
    usagers = normaliser(usagers)

    print("\nTailles brutes :")
    print("caracteristiques:", caracteristiques.shape)
    print("lieux:", lieux.shape)
    print("vehicules:", vehicules.shape)
    print("usagers:", usagers.shape)

    print("\n2. Jointures principales")

    # USAGERS ← VEHICULES (Num_Acc + id_vehicule)
    data1 = usagers.merge(
        vehicules,
        on=["num_acc", "id_vehicule", "annee"],
        how="left",
        suffixes=("_usag", "_veh")
    )

    # + CARACTERISTIQUES (Num_Acc)
    data2 = data1.merge(
        caracteristiques,
        on=["num_acc", "annee"],
        how="left"
    )

    # + LIEUX (Num_Acc)
    master = data2.merge(
        lieux,
        on=["num_acc", "annee"],
        how="left"
    )

    print("\nRépartition par année :")
    print(master["annee"].value_counts().sort_index())

    # =========================
    # EXPORT FINAL
    # =========================
    output = project_dir / "BAAC_MASTER_2019_2024.csv"
    master.to_csv(output, sep=";", index=False, encoding="utf-8")

    print(f"\n💾 Sauvegardé : {output}")
    print("Colonnes finales :", len(master.columns))

    print("\nAperçu :")
    print(master[["num_acc", "id_vehicule", "annee"]].head(5))


if __name__ == "__main__":
    main()

import os
import geopandas as gpd
import pandas as pd
import matplotlib.pyplot as plt

# Fonction pour vérifier la présence des fichiers nécessaires
def check_shapefile_files(shapefile_path):
    base_path, _ = os.path.splitext(shapefile_path)
    extensions = ['.shp', '.shx', '.dbf', '.prj']
    missing_files = [base_path + ext for ext in extensions if not os.path.exists(base_path + ext)]
    if missing_files:
        print(f"Fichiers manquants : {missing_files}")
    else:
        print("Tous les fichiers nécessaires sont présents.")

# Chemin vers le shapefile
shapefile_path = '/Users/vignaulucas/Desktop/stikom_models/stikomModels/dataNSIS/Soils25K_v10d.shp'

# Vérifier la présence des fichiers nécessaires
check_shapefile_files(shapefile_path)

# Charger le shapefile
gdf = gpd.read_file(shapefile_path)

# Afficher les premières 20 lignes
print(gdf.head(20))

# Convertir en DataFrame pandas
df = pd.DataFrame(gdf)

# Chemin vers le fichier CSV
csv_file_path = '/Users/vignaulucas/Desktop/stikom_models/stikomModels/dataNSIS/soil_data.csv'

# Enregistrer dans un fichier CSV
df.to_csv(csv_file_path, index=False)

print(f"Les données ont été enregistrées dans le fichier {csv_file_path}")

# # Afficher le shapefile
# gdf.plot(column='pH_W_Med', legend=True, cmap='viridis')
# plt.title('Distribution du pH du sol')
# plt.show()

# for i, (npk, ph, pred_npk, pred_ph) in results.iterrows():
#         print(f"Sample {i+1}: Real NPK: {npk:.2f}, Real pH: {ph:.2f}, Predicted NPK: {pred_npk:.2f}, Predicted pH: {pred_ph:.2f}")

#     # Extracting and displaying rules from one of the trees in the Random Forest
#     tree = rf_model.estimators_[0]
#     export_graphviz(tree, out_file='tree.dot', feature_names=['h', 'V', 'C'], filled=True, rounded=True, special_characters=True)
    
#     with open("tree.dot") as f:
#         dot_graph = f.read()
    
#     graphviz.Source(dot_graph).view()
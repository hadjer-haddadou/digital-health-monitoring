import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

data = pd.read_csv('dataset.csv')

# Renommage des colonnes pour plus de clarté
data = data.rename(columns={
    'sr': 'Snoring_Rate', 
    'rr': 'Respiration_Rate', 
    't': 'Temperature', 
    'lm': 'Limb_Movement', 
    'bo': 'Blood_Oxygen', 
    'rem': 'REM_Sleep', 
    'sr.1': 'Sleep_Hours',
    'hr': 'Heart_Rate', 
    'sl': 'Stress_Level'
})

#print(data.head())
# 1. Compter le nombre de "trous" (valeurs manquantes) par colonne
print("--- Valeurs manquantes ---")
print(data.isnull().sum())

# 2. Vérifier que Python comprend bien que ce sont des nombres (float/int) et pas du texte (object)
print("\n--- Types de données ---")
print(data.info())

# 3. Afficher un résumé clinique (moyenne, min, max) pour repérer d'éventuelles valeurs aberrantes (ex: un rythme cardiaque à 500)
print("\n--- Statistiques descriptives ---")
print(data.describe())

#ratio evaluant  le comportement du systeme nerveux autonome , donc biomarqueur physiopathologique 
data['Cardio_Resp_Ratio'] = data['Heart_Rate'] / data['Respiration_Rate']

#print("\n--- Nouvel indice créé ---")
#print(data[['Heart_Rate', 'Respiration_Rate', 'Cardio_Resp_Ratio']].head())

#un adulte au repos présente un ratio cardio-respiratoire naturel situé entre 3.0 et 5.0
# en cas de stress aigu l'hyperventilation fait chuter le ratio , une tachycardie isolée le fait exploser 

#indicateur de la qualité réparatrice du sommeil
data['REM_Density'] = data['REM_Sleep'] / data['Sleep_Hours']

#print("\n--- Nouvel indice créé ---")
#print(data[['REM_Density']].head())

#score qui capte l'agitation nocturne souvent liée au stress physiologique 
data['Physical_Distress'] = data['Limb_Movement'] * data['Snoring_Rate']

#print("\n--- Nouvel indice créé ---")
#print(data[['Physical_Distress']].head())

print(data[['Cardio_Resp_Ratio', 'REM_Density', 'Physical_Distress']].head())


# Génération de la boîte à moustaches
#sns.boxplot(data=data, x='Stress_Level', y='Cardio_Resp_Ratio')
#plt.show()





# ==========================================
# ENQUÊTE CLINIQUE : ANOMALIE DU GROUPE 3
# ==========================================

# 1. On filtre uniquement les patients dont le niveau de stress est 3
groupe_3 = data[data['Stress_Level'] == 3]

# 2. Parmi eux, on isole ceux dont la densité calculée est aberrante (ex: > 100)
anomalies = groupe_3[groupe_3['REM_Density'] > 100]

# 3. On affiche la donnée brute pour comprendre le bug
print("\n--- Lignes défectueuses (Stress = 3) ---")
print(anomalies[['REM_Sleep', 'Sleep_Hours', 'REM_Density']].head(10))


#des valeurs aberantes surement du a un problem de mesure ou d'enregisrement exemple sujet numéro 99 qui a eu 0 h de sommeil mais un score REM de 95
#donc filtrer et nettoyer 

# ==========================================
# NETTOYAGE DES ARTEFACTS MATÉRIELS
# ==========================================
# Suppression des erreurs de capteurs (temps de sommeil physiologiquement impossible)
data = data[data['Sleep_Hours'] >= 1.0]

print(f"\nDonnées nettoyées. Il reste {len(data)} patients valides.")


# Création de la grille (1 ligne, 3 colonnes)
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Graphique 1 : Ratio Cardio-Respiratoire (Boîte + Points)
sns.boxplot(data=data, x='Stress_Level', y='Cardio_Resp_Ratio', ax=axes[0], color='lightgray')
sns.stripplot(data=data, x='Stress_Level', y='Cardio_Resp_Ratio', ax=axes[0], alpha=0.5, jitter=True)
axes[0].set_title('Ratio Cardio-Respiratoire')

# Graphique 2 : Densité du Sommeil Paradoxal (Boîte + Points)
sns.boxplot(data=data, x='Stress_Level', y='REM_Density', ax=axes[1], color='lightgray')
sns.stripplot(data=data, x='Stress_Level', y='REM_Density', ax=axes[1], alpha=0.5, jitter=True)
axes[1].set_title('Densité du Sommeil Paradoxal')

# Graphique 3 : Indice de Détresse Physique (Boîte + Points)
sns.boxplot(data=data, x='Stress_Level', y='Physical_Distress', ax=axes[2], color='lightgray')
sns.stripplot(data=data, x='Stress_Level', y='Physical_Distress', ax=axes[2], alpha=0.5, jitter=True)
axes[2].set_title('Indice de Détresse Physique')

plt.tight_layout()
plt.show()

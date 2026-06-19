Besoin du client 4: Prédiction de la puissance nominale d'une installation

bibliothèque joblib: sert à importer le modèle qui est contenu dans le fichier "modèle_puissance.pkl"

objet OrdinalEncoder de la bibliothèque sklearn: sert à convertir les données textuelles au formal numérique en préparation de la prédiction

bibliothèque pandas: sert à mettre les données au format dataframe, utilisable par l'outil

fonction "predict": prépare les données en les mettant en format numérique et sélectionnant les colonnes nécessaire puis effectue la prédiction de la puissance nominale

Guide d'utilisation:

1) S'assurer d'avoir importé toutes les bibliothèques
2) Renseigner les données de l'installation voulue sous forme de dataframe, un objet "data" est déjà fourni à titre de test
3) Mettre ces données comme paramètre de la fonction "predict" et l'appeler
4) exécuter le script
5) la puissance prédite est ensuite retournée par la fonction et peu notamment être affichée par un print
  
![](images/fasest.png)

Faculté des Sciences Économiques, Sociales et des Territoires \- Université de Lille 

| Projet de MLOps Accidents de la circulation routière en France |
| :---: |

Année universitaire 2025 \- 2026

Travail réalisé par : 

Astor DE LA PUENTE / Arissara KONGHUAIROB  
Benoît LHERMITTE / Rydch Junior VINCENT 

Master 2 SIAD \- DS

# SOMMAIRE 

[SOMMAIRE](#sommaire)

[Introduction](#introduction)

[I. Exploration et préparation de la base de données](#i-exploration-et-préparation-de-la-base-de-données)


[1\. Description de la base de données](#description-de-la-base-de-données)

[2\. Étude des valeurs manquantes](#étude-des-valeurs-manquantes)

[3\. Traitement des outliers et anomalies](#traitement-des-outliers-et-anomalies)

[II. Analyse et feature engineering](#ii-analyse-et-feature-engineering)

[1\. Analyse descriptive et profilage](#analyse-descriptive-et-profilage)

[2\. Feature engineering et sélection stratégique](#feature-engineering-et-sélection-stratégique)

[III. Modélisation prédictive et analyse des résultats](#iii-modélisation-prédictive-et-analyse-des-résultats)

[1\. Modélisation et Performances](#modélisation-et-performances)

[2\. Facteurs Influents de Gravité](#facteurs-influents-de-gravité)

[3. Interprétation des coefficients](#interprétation-des-coefficients)


[Conclusion](#conclusion)



# Introduction

Notre projet s'articule autour d'un enjeu majeur de santé publique : la sécurité routière. Pour mener cette étude, nous exploitons le Fichier National des Accidents Corporels de la Circulation (BAAC). Ce registre est alimenté par les rapports des forces de l'ordre qui interviennent sur le lieu de l’accident (procès-verbaux et rapports simplifiés) et constitue la référence statistique de l'accidentalité en France.

Notre jeu de données est axé sur la période 2019-2024, ce qui permet d'intégrer les évolutions récentes des modes de transport (essor de certains types de mobilités comme la trottinette électrique par exemple) tout en tenant compte des variations de flux liées aux crises sanitaires passées.

Ce jeu de données est éclatée en quatre tables complémentaires, que nous devons fusionner pour obtenir une vision d’ensemble de chaque événement :

* La table Caractéristiques : Circonstances temporelles, conditions atmosphériques, type d'intersection et luminosité.  
* La table Lieux : Réseau routier (autoroute, départementale), numéro de voie de la route, régime de circulation…  
* La table Véhicules : Catégorie du véhicule, point de choc initial et manœuvre effectuée.  
* La table Usagers : Profil des victimes (âge, sexe), équipement de sécurité (ceinture, casque) et gravité de l'état de santé.


L’objectif central de notre projet est de concevoir, grâce à l’analyse de nos différentes variables un modèle de machine learning robuste et performant capable de prédire, en temps réel, la gravité d'un accident dès son signalement. Le modèle devra classer l'accident selon un score de criticité (Léger vs Grave/Mortel).

Grâce à ce modèle, on pourra dans les minutes qui suivent l’accident, optimiser l’allocation des ressources. Le modèle permettra d’identifier instantanément si l'envoi de moyens lourds (SMUR, hélicoptère) est nécessaire, et d’ajuster l'effectif dépêché sur place en fonction de la probabilité de présence de victimes incarcérées ou en urgence absolue. Enfin, notre modèle pourra permettre l’identification de facteurs récurrents de gravité des accidents de la route afin de suggérer des aménagements d'infrastructure aux pouvoirs publics.

## I. Exploration et préparation de la base de données

1. ### Description de la base de données 

La base fournie contient des informations sur un ensemble d’accidents, chacun étant décrit par plusieurs variables correspondant à des caractéristiques propres à ce dernier. Chaque colonne représente une caractéristique de l’accident, et chaque ligne correspond à un accident.

Voici le dictionnaire de données initiales de notre base de données :  
   
Table Caractéristiques : 

| Variable | Description | Modalités Exemples |
| :---- | :---- | :---- |
| an / mois / jour | Date précise de l'accident. | *AAAA / MM / JJ* |
| hrmn | Heure et minutes de l'accident. | *HH:MM* |
| lum | Conditions de luminosité. | Plein jour, Nuit avec ou sans éclairage |
| atm | Conditions atmosphériques. | Normale, Pluie, Neige, Brouillard, Vent fort |
| int | Type d'intersection. | Hors intersection, Giratoire, Intersection en X, T ou Y |
| col | Type de collision. | Frontale, Par l'arrière, Chocs multiples, Sans collision |
| dep / com | Localisation géographique (Code INSEE). | Département et commune |
| lat / long | Coordonnées GPS (Projection WGS84). | Données de géolocalisation précises |

Table Lieux : 

| Variable | Description | Modalités Exemples |
| :---- | :---- | :---- |
| catr | Catégorie de route. | Autoroute, Nationale, Départementale, Voie communale |
| vma | Vitesse maximale autorisée. | 30, 50, 80, 110, 130 km/h |
| circ | Régime de circulation. | Sens unique, Bidirectionnelle, Chaussées séparées |
| prof | Profil en long (déclivité). | Plat, Pente, Sommet de côte |
| plan | Tracé en plan. | Rectiligne, Courbe (gauche/droite), En "S"  |
| surf | État de la surface. | Normale, Mouillée, Verglacée, Corps gras |
| infra | Aménagement particulier. | Tunnel, Pont, Passage à niveau, Zone piétonne |

Table Usagers : 

| Variable | Description | Modalités Exemples |
| :---- | :---- | :---- |
| catu | Catégorie d'usager. | Conducteur, Passager, Piéton |
| sexe | Sexe de l'usager. | Masculin, Féminin |
| an\_nais | Année de naissance. | Permet de calculer l'âge de la victime |
| trajet | Motif du déplacement. | Domicile-travail, Loisirs, Professionnel |
| secu1/2/3 | Équipements de sécurité utilisés | Ceinture, Casque, Gants, Gilet réfléchissant  |
| grav | Variable Cible (Target) : Gravité. | Indemne, Tué, Blessé hospitalisé, Blessé léger |

Table Véhicules : 

| Variable | Description | Modalités Exemples |
| :---- | :---- | :---- |
| catv | Catégorie du véhicule. | VL, PL, Vélo, EDP (trottinettes), Moto, Tramway  |
| motor | Type de motorisation. | Hydrocarbures, Électrique, Hybride, Hydrogène |
| manv | Manœuvre principale avant choc. | Sans changement, Dépassement, Demi-tour, Contresens |
| choc | Point de choc initial. | Avant, Arrière, Côté droit/gauche, Chocs multiples |
| obs / obsm | Obstacle heurté (fixe ou mobile). | Arbre, Véhicule, Piéton, Animal sauvage |

La base de données regroupant les quatre tables sur la période 2019-2024, comporte 820413 observations et 58 variables.

2. ### Étude des valeurs manquantes

L'analyse a révélé des disparités importantes dans le taux de remplissage des variables :  
Certaines variables ont en proportion beaucoup de valeurs manquantes comme lartpc (largeur du terre-plein central), occutc (nombre d'occupants TC) et v2 (indice alphanumérique de la route).  
Ces variables présentant plus de 90% de valeurs manquantes ont été supprimées ou transformées (ex: lartpc transformée en indicateur binaire de présence de terre-plein).

3. ### Traitement des outliers et anomalies

La méthode de l'Écart Interquartile (IQR) a permis de relever des valeurs aberrantes. Concernant la vitesse, 25764 cas à 0 km/h et 225 cas à plus de 130 km/h (allant jusqu'à 901\) ont été identifiés. Ces valeurs ont été imputées par la médiane (50 km/h). Pour l’âge, 201 cas d'âges extrêmes (\> 110 ans) ont été détectés via an\_nais.

## II. Analyse et feature engineering 

1. ### Analyse descriptive et profilage 

Notre variable cible de gravité de l’accident se répartit de cette façon (après suppressions des valeurs manquantes) : 

* Indemne : 42,45% (348 299 usagers).  
* Blessé léger : 39,90% (327 378 usagers).  
* Blessé hospitalisé : 15,04% (123 399 usagers).  
* Tué : 2,55% (20 880 usagers).

Lors des accidents, la population est majoritairement masculine (551 793 hommes contre 256 335 femmes) avec un âge médian de 38 ans. Les accidents ont souvent lieu en milieu urbain, la vitesse maximale autorisée (VMA) la plus fréquente est 50 km/h (397 431 cas), bien que la moyenne s'élève à 58,4 km/h.

2. ### Feature engineering et sélection stratégique 

Afin d’augmenter les performances de notre future modèle, certaines variables ont été transformés dans l’optique d’être plus adapté à l’apprentissage : 

* Conversion de an\_nais en age. C'est une variable critique car la vulnérabilité physique aux chocs est non linéaire.  
* Création de l'indicateur binaire presence\_bande\_cyclable à partir de lartpc pour simplifier l'analyse des aménagements de sécurité.  
* Regroupement de la variable cible (grav) : Transformation des 4 modalités d'origine en deux classes binaires, selon un critère d’état de la victime:   
  * Classe 0 (Léger) : Indemne (1) et Blessé léger (4).  
  * Classe 1 (Grave) : Tué (2) et Blessé hospitalisé (3). Environ 35,8% des accidents de la base  
* Suppression de variables :  
  * Suppression des identifiants techniques (num\_acc, id\_vehicule, id\_usager) jugés inutiles pour notre modèle.  
  * Suppression des données textuelles (adr, voie) trop hétérogènes pour un modèle standard.

Nous avons créé des indicateurs permettant une simplification de la documentation pour les témoins de l'accident et les secours ou force de l’ordre présentes sur place :

### 

* Profils Usagers : Nous avons généré le nombre total de personnes impliqués (nb\_usager) et des indicateurs de présence (0 ou 1\) pour les populations vulnérables : Piétons , Enfants (\< 14 ans) et Seniors (plus de 65 ans) basés sur l'année de naissance. Signaler la présence d'une seule personne fragile suffit souvent à faire basculer l'accident dans la catégorie "Grave".

* Familles de Véhicules : Regroupement des codes catv en 7 catégories métier (Vélos, 2RM, VL/VU, Poids Lourds, etc.).

Nous avons également transformé l’heure et le jour en fonctions Sinus/Cosinus pour modéliser la périodicité réelle des flux (nuit/jour).

## 

## III. Modélisation prédictive et analyse des résultats

1. ### Modélisation et Performances

Le modèle a été entraîné sur la période 2019-2023 et évalué sur l'année 2024 pour garantir une capacité de généralisation réelle (test en conditions "futures").  
Nous avons testé trois approches via des pipelines de prétraitement automatisés :

| Modèle | ROC-AUC | Précision (Grave) | Rappel (Grave) |
| :---- | :---- | :---- | :---- |
| Régression Logistique | 0,789 | 0,64 | 0,57 |
| Random Forest | 0,776 | 0,72 | 0,35 |
| Gradient Boosting | 0,792 | 0,69 | 0,45 |

### 

Dans ce projet, le Rappel est la métrique la plus critique car elle mesure la capacité du modèle à identifier correctement tous les accidents réellement graves. Un faible rappel signifierait que le modèle classerait comme "Légers" des accidents qui sont en réalité "Graves". Pour les secours, cela se traduirait par un envoi insuffisant de moyens alors que des vies sont en jeu. En abaissant le seuil de décision à 0,4, nous acceptons qu’il y ait un peu plus de fausses alertes pour garantir que 68,3 % des cas graves soient immédiatement détectés. Il est préférable d'envoyer des moyens lourds par précaution plutôt que de sous-estimer une urgence vitale. Dans notre cas, le modèle identifie près de 7 accidents graves sur 10 dès le signalement, permettant un déclenchement anticipé des moyens lourds (SMUR, hélicoptère).  
La Régression Logistique a été privilégiée face au Gradient Boosting car elle offre un meilleur équilibre des performances sur la classe cible, affichant un rappel initial de 0,57 contre 0,35 pour la Random Forest malgré un score ROC-AUC très proche de 0,789. Ce choix se justifie également par l'interprétabilité du modèle qui permet d'expliquer les prédictions aux décideurs publics en isolant l'impact de facteurs critiques tels que la vitesse maximale autorisée ou la vulnérabilité liée à l'âge. En évitant l'opacité des modèles complexes, les coefficients facilitent la compréhension des causes de gravité, qu'il s'agisse d'un décès ou d'un blessé hospitalisé plus de 24 heures.

Sur le plan opérationnel, la Régression Logistique constitue un modèle léger et rapide, idéal pour une intégration en temps réel dès qu'un accident corporel est recensé par les forces de l'ordre sur une voie ouverte à la circulation. De plus, cette méthode renforce la confiance des utilisateurs métier, comme les régulateurs du SAMU, en rendant explicites les critères de criticité d'un accident au contraire de certains modèles faisant office de boîtes noires.

2. ### Facteurs Influents de Gravité

L’analyse de l’importance des variables identifie la vitesse maximale autorisée comme le premier levier de mortalité routière, car la vitesse transforme ce qui aurait pu être un accident léger en un événement dramatique. Cette dangerosité est intrinsèquement liée à la structure du réseau routier, où les routes départementales et nationales s'avèrent plus propices aux accidents que les autoroutes ou les voies urbaines. Parallèlement, le profil des usagers joue un rôle déterminant dans le score de criticité, puisque la présence de seniors ou de conducteurs de deux-roues motorisés augmente drastiquement la probabilité de blessures graves ou de décès en raison de leur vulnérabilité physique. Enfin, la configuration mécanique du choc confirme que les collisions frontales demeurent les plus létales pour les occupants, surpassant largement en gravité les chocs par l'arrière.

3. ###  Interprétation des coefficients

Pour rendre les résultats de notre modèle interprétables, nous convertissons les coefficients en Odds Ratios (OR) grâce à la fonction exponentielle. Un OR supérieur à 1 indique que la modalité augmente les chances de gravité par rapport à la référence, tandis qu'un OR inférieur à 1 indique un effet protecteur ou une diminution de la probabilité de gravité.

Le modèle révèle des disparités géographiques majeures. Par exemple, le département 70 affiche un coefficient de 1,83, soit un OR de 6,23. Cela signifie qu'à caractéristiques identiques, un accident survenant dans ce département a 6,2 fois plus de chances d'être grave qu'un accident dans le département de référence. À l'inverse, le département 92 (beta \= \-2,08, OR \= 0,13) réduit ces chances de 87 %. Ces écarts reflètent souvent des différences de typologie de réseau et de vitesse de circulation locale.

Également, le contexte "hors réseau public" (catr\_5) présente un OR de 2,20, ce qui signifie qu'un accident y est 2,2 fois plus susceptible d'être grave qu'en agglomération classique. À l'opposé, l'autoroute (catr\_1) présente un OR de 0,57, réduisant les chances de gravité de 43 %, probablement grâce à la séparation des flux et l'absence d'obstacles latéraux.

De plus, la présence de boue (surf\_6) sur la chaussée multiplie les chances de gravité par 2,38. De manière plus surprenante, les routes enneigées (surf\_5, OR \= 0,44) ou avec corps gras (surf\_8, OR \= 0,52) sont associées à une baisse de la probabilité de gravité de près de 50 %. Ce résultat suggère une adaptation du comportement des usagers qui réduisent leur vitesse face à un danger visible.

Enfin, le type de collision influe également sur la gravité de l’accident. Les collisions par l'arrière (col\_2) et en chaîne (col\_4) réduisent les chances de gravité de respectivement 44 % et 51 % par rapport aux collisions frontales, confirmant que ces dernières absorbent une énergie cinétique bien plus létale.

## **Conclusion**

Ce projet démontre l'efficacité de la science des données pour répondre à des problématiques bien réelles. En binarisant la gravité pour isoler les accidents impliquant des personnes tuées ou blessées hospitalisées (plus de 24h), nous fournissons un score de criticité des accidents directement utile aux services de secours.

Le choix de la Régression Logistique, avec un seuil de décision ajusté à 0,4, permet d'obtenir un rappel de 68,3 %. Dans un contexte de sécurité routière, le rappel est la métrique prioritaire : il vaut mieux générer quelques fausses alertes (envoyer des moyens pour un accident finalement léger) que de rater un accident grave. Ce modèle permet d'identifier près de 7 accidents graves sur 10 dès le premier appel, facilitant le déclenchement immédiat de moyens lourds (SMUR, hélicoptère).

L'interprétation des coefficients souligne que la vitesse maximale autorisée (VMA) et la typologie des routes (départementales vs autoroutes) sont les leviers d'action fondamentaux. Les résultats suggèrent que les politiques publiques doivent privilégier les aménagements séparant les flux de circulation pour limiter les chocs frontaux, et maintenir une vigilance accrue sur les réseaux secondaires où la gravité des accidents est plus importante.  

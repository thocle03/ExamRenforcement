# Apprentissage par Renforcement - Tennis Atari

## 📋 Présentation du Projet

Ce projet implémente et compare trois algorithmes d'apprentissage par renforcement (DQN, PPO, A2C) sur l'environnement **ALE/Tennis-v5** d'Atari.

![tennis](https://github.com/user-attachments/assets/9eb7acf7-23fe-4648-ad10-53419ccb9336)

---

## 1. L'Environnement Tennis (Atari)

### Description de l'environnement
**Tennis** est un jeu Atari classique où l'agent contrôle une raquette pour renvoyer une balle de tennis. L'objectif est de marquer des points en faisant rebondir la balle du côté adverse sans la manquer.

### Caractéristiques techniques
- **Type d'observation** : Images RGB (frames du jeu)
- **Prétraitement** : 
  - Redimensionnement à 84×84 pixels
  - Conversion en niveaux de gris
  - Stack de 4 frames consécutives (pour capturer le mouvement)
  - Frame skipping (pour réduire la redondance temporelle)
- **Espace d'observation final** : `(4, 84, 84)` - 4 frames empilées de 84×84 pixels
- **Récompenses** : 
  - Points positifs quand l'agent marque
  - Points négatifs quand l'adversaire marque
  - L'objectif est de maximiser le score cumulé

### Actions disponibles
L'agent peut effectuer **18 actions discrètes** dans l'environnement Tennis :

| Action | Description |
|--------|-------------|
| 0 | NOOP (Pas d'action) |
| 1 | FIRE (Lancer la balle) |
| 2-17 | Combinaisons de mouvements (Haut/Bas/Gauche/Droite + FIRE) |

Les actions principales sont :
- **Déplacements** : Haut, Bas, Gauche, Droite
- **FIRE** : Frapper la balle
- **Combinaisons** : Mouvements + FIRE simultanément

L'agent doit apprendre à :
1. Positionner sa raquette correctement
2. Anticiper la trajectoire de la balle
3. Frapper au bon moment
4. Renvoyer la balle vers l'adversaire

---

## 2. Algorithmes d'Apprentissage par Renforcement

Nous comparons trois algorithmes state-of-the-art pour ce problème :

### DQN (Deep Q-Network)

#### Pourquoi DQN ?
DQN est un algorithme **off-policy** basé sur la Q-learning qui a révolutionné l'apprentissage par renforcement en 2015 en atteignant des performances humaines sur plusieurs jeux Atari. Il est particulièrement adapté pour :
- Les espaces d'actions discrets (comme Tennis avec 18 actions)
- Les observations visuelles complexes (grâce au CNN)
- L'apprentissage à partir d'expériences passées

#### Paramètres détaillés

```python
DQN(
    policy="CnnPolicy",              # Politique basée sur CNN pour traiter les images
    learning_rate=1e-4,              # Taux d'apprentissage faible pour stabilité
    buffer_size=20_000,              # Taille du replay buffer (mémoire d'expériences)
    learning_starts=2_000,           # Commence à apprendre après 2000 steps d'exploration
    batch_size=32,                   # Nombre d'expériences par mise à jour
    gamma=0.99,                      # Facteur d'actualisation (importance du futur)
    train_freq=4,                    # Mise à jour tous les 4 steps
    gradient_steps=1,                # 1 étape de gradient par update
    target_update_interval=10_000,   # Mise à jour du réseau cible tous les 10k steps
    exploration_fraction=0.20,       # 20% du temps pour diminuer l'exploration
    exploration_final_eps=0.01,      # Epsilon minimal (1% d'exploration aléatoire)
)
```

**Explication des paramètres clés :**
- **CnnPolicy** : Réseau de neurones convolutionnel pour traiter les images 84×84
- **buffer_size** : Stocke 20 000 transitions (état, action, récompense, état suivant)
- **learning_starts** : Accumule de l'expérience avant d'apprendre (évite l'overfitting initial)
- **gamma=0.99** : Valorise fortement les récompenses futures (stratégie long-terme)
- **target_update_interval** : Réseau cible stable pour réduire la variance de l'apprentissage
- **exploration** : Stratégie ε-greedy qui diminue de 1.0 à 0.01 sur 20% de l'entraînement

**Configuration environnement :**
- **1 environnement** : DQN est off-policy, il apprend depuis le replay buffer

---

### PPO (Proximal Policy Optimization)

#### Pourquoi PPO ?
PPO est un algorithme **on-policy** moderne et robuste, considéré comme l'un des meilleurs algorithmes policy gradient. Il est excellent pour :
- La stabilité d'apprentissage (clip des mises à jour)
- L'efficacité computationnelle
- La fiabilité sur une grande variété de tâches

#### Paramètres détaillés

```python
PPO(
    policy="CnnPolicy",              # Politique basée sur CNN pour traiter les images
    learning_rate=2.5e-4,            # Taux d'apprentissage standard pour PPO
    n_steps=128,                     # Nombre de steps par rollout
    batch_size=128,                  # Taille des mini-batches pour l'optimisation
    n_epochs=4,                      # Nombre de passes sur les données collectées
    gamma=0.99,                      # Facteur d'actualisation
    gae_lambda=0.95,                 # Lambda pour Generalized Advantage Estimation
    clip_range=0.1,                  # Clip pour limiter les changements de politique
    ent_coef=0.01,                   # Coefficient d'entropie (encourage l'exploration)
    vf_coef=0.5,                     # Coefficient de la value function loss
    max_grad_norm=0.5,               # Clipping du gradient pour stabilité
)
```

**Explication des paramètres clés :**
- **n_steps=128** : Collecte 128 transitions avant chaque mise à jour
- **n_epochs=4** : Réutilise 4 fois les données collectées (efficacité d'échantillonnage)
- **gae_lambda=0.95** : Compromis entre biais et variance pour estimer l'advantage
- **clip_range=0.1** : Limite les changements drastiques de politique (stabilité)
- **ent_coef=0.01** : Bonus d'entropie pour éviter la convergence prématurée
- **vf_coef=0.5** : Équilibre entre optimisation de la value function et de la policy
- **max_grad_norm=0.5** : Empêche les gradients explosifs

**Configuration environnement :**
- **1 environnement** : PPO fonctionne bien avec un seul environnement pour Atari

---

### A2C (Advantage Actor-Critic)

#### Pourquoi A2C ?
A2C est la version **synchrone** de A3C, un algorithme actor-critic qui combine les avantages des méthodes basées sur la valeur et sur la politique. Il est particulièrement adapté pour :
- L'apprentissage parallèle multi-environnements
- La convergence rapide grâce aux mises à jour fréquentes
- L'efficacité avec plusieurs workers synchrones

#### Paramètres détaillés

```python
A2C(
    policy="CnnPolicy",              # Politique basée sur CNN pour traiter les images
    learning_rate=7e-4,              # Taux d'apprentissage élevé pour A2C
    n_steps=8,                       # Nombre de steps avant mise à jour (très fréquent)
    gamma=0.99,                      # Facteur d'actualisation
    gae_lambda=0.95,                 # Lambda pour GAE (réduit variance)
    ent_coef=0.01,                   # Coefficient d'entropie (exploration)
    vf_coef=0.25,                    # Coefficient de la value function (réduit vs PPO)
    max_grad_norm=0.5,               # Clipping du gradient
    rms_prop_eps=1e-5,               # Epsilon pour RMSprop (stabilité numérique)
    use_rms_prop=True,               # Utilise RMSprop au lieu d'Adam
    normalize_advantage=True,        # Normalise l'advantage (stabilité)
)
```

**Explication des paramètres clés :**
- **n_steps=8** : Mises à jour très fréquentes (avec 4 envs = 32 transitions par update)
- **learning_rate=7e-4** : Plus élevé que DQN/PPO car mises à jour plus fréquentes
- **gae_lambda=0.95** : Réduit la variance par rapport à λ=1.0
- **vf_coef=0.25** : Moins de poids sur la value function pour éviter l'overfitting
- **use_rms_prop=True** : RMSprop est l'optimiseur classique pour A2C/A3C
- **normalize_advantage=True** : Normalise les advantages pour stabiliser l'apprentissage
- **rms_prop_eps=1e-5** : Évite la division par zéro dans RMSprop

**Configuration environnement :**
- **4 environnements parallèles** : A2C est conçu pour l'apprentissage multi-environnements synchrone
- Collecte des expériences de 4 workers en parallèle
- Améliore la diversité des données et réduit la corrélation
- Accélère significativement la convergence

---

## 3. Configuration Expérimentale

### Paramètres communs
- **Total timesteps** : 300 000 steps d'entraînement
- **Seeds** : 3 seeds différents (0, 1, 2) pour la robustesse statistique
- **Évaluation** : 
  - DQN/PPO : Tous les 10 000 steps
  - A2C : Tous les 50 000 steps
  - 3 épisodes par évaluation
  - 5 épisodes pour l'évaluation finale

### Preprocessing Atari
- **AtariWrapper** : Prétraitement standard Atari (grayscale, resize, frame skip)
- **Frame stacking** : 4 frames consécutives empilées
- **Frame shape** : 84×84 pixels

### Environnements spécialisés
```python
# DQN : 1 environnement (off-policy avec replay buffer)
make_vec_env_dqn(seed=seed, n_stack=4)

# PPO : 1 environnement (on-policy avec rollout buffer)
make_vec_env_ppo(seed=seed, n_stack=4)

# A2C : 4 environnements parallèles (multi-worker synchrone)
make_vec_env_a2c(seed=seed, n_envs=4, n_stack=4)
```

---

## 4. Résultats et Visualisation

Le notebook génère :
1. **Courbes d'apprentissage** : Performance moyenne ± écart-type sur les 3 seeds
2. **Tableau récapitulatif** : Moyenne et variance des performances finales par algorithme
3. **Logs détaillés** : Temps d'entraînement, évaluations intermédiaires

---

## 5. Utilisation

### Installation
```python
%pip install "gymnasium[atari,accept-rom-license]" stable-baselines3[extra] ale-py shimmy
```

### Entraînement
Exécutez les cellules du notebook dans l'ordre :
1. Installation des dépendances
2. Imports et configuration
3. Définition des environnements
4. Entraînement des modèles
5. Visualisation des résultats

### Structure du code
```
tennis (1).ipynb
├── Setup & imports
├── Configuration (ENV_ID, TIMESTEPS, SEEDS)
├── Fonctions de création d'environnement (spécialisées par algo)
├── Callback d'évaluation avec barre de progression
├── Fonctions d'entraînement (train_dqn, train_ppo, train_a2c)
├── Boucle d'entraînement principale
└── Visualisation et analyse des résultats
```

---

## 6. Comparaison des Algorithmes

| Algorithme | Type | Environnements | Sample Efficiency | Stabilité | Vitesse |
|------------|------|----------------|-------------------|-----------|---------|
| **DQN** | Off-policy | 1 | ⭐⭐⭐ | ⭐⭐ | ⭐⭐ |
| **PPO** | On-policy | 1 | ⭐⭐ | ⭐⭐⭐ | ⭐⭐ |
| **A2C** | On-policy | 4 | ⭐⭐ | ⭐⭐ | ⭐⭐⭐ |

**Légende :**
- **Sample Efficiency** : Capacité à apprendre avec peu de données
- **Stabilité** : Fiabilité de la convergence
- **Vitesse** : Rapidité d'entraînement (wall-clock time)

---

## Références

- **DQN** : Mnih et al., "Human-level control through deep reinforcement learning", Nature 2015
- **PPO** : Schulman et al., "Proximal Policy Optimization Algorithms", 2017
- **A2C/A3C** : Mnih et al., "Asynchronous Methods for Deep Reinforcement Learning", 2016
- **Stable-Baselines3** : https://stable-baselines3.readthedocs.io/

---

<<<<<<< HEAD
## 👤 Auteur
Hexa Team
Projet d'apprentissage par renforcement sur l'environnement Tennis Atari.

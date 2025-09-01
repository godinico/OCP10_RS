import azure.functions as func # type: ignore
import pickle
import json
import logging
import os
from azure.storage.blob import BlobServiceClient
from typing import Optional

# Initialisation de l'application Azure Functions
app = func.FunctionApp()

# ════════════════════════════════════════════════════════════
# ──────────────────── CHARGEMENT DU MODÈLE ──────────────────
# ════════════════════════════════════════════════════════════
MODEL = None
try:
    # Récupération de la chaîne de connexion au Blob Storage depuis les variables d'environnement
    conn_str = os.environ.get("BLOB_STORAGE_CONNECTION")
    if not conn_str:
        raise ValueError("La variable d'environnement BLOB_STORAGE_CONNECTION n'est pas définie")
    
    logging.info("Chargement du modèle SVD depuis Azure Blob Storage...")
    blob_service = BlobServiceClient.from_connection_string(conn_str)
    container_client = blob_service.get_container_client("recommendation-models")
    blob_client = container_client.get_blob_client("svdpp_model.pkl")
    blob_data = blob_client.download_blob().readall()
    
    # Désérialisation du modèle
    MODEL = pickle.loads(blob_data)
    logging.info(f"Modele charge : {len(MODEL.trainset.all_users())} utilisateurs, {len(MODEL.trainset.all_items())} items")
except Exception as e:
    logging.error(f"Echec du chargement du modèle : {str(e)}")
    MODEL = None

# ════════════════════════════════════════════════════════════
# ──────────────── FONCTION DE RECOMMANDATION ────────────────
# ════════════════════════════════════════════════════════════
@app.function_name(name="RecommendationFunction")
@app.route(route="recommendations", methods=["GET", "POST"])
def recommendation_function(req: func.HttpRequest) -> func.HttpResponse:
    """
    Point d'entrée HTTP pour générer des recommandations d'articles à partir d'un modèle SVD.
    
    Args:
        req (func.HttpRequest): Requête HTTP contenant les paramètres utilisateur.
    
    Returns:
        func.HttpResponse: Réponse HTTP contenant les recommandations ou une erreur.
    """
    global MODEL
    if MODEL is None:
        return create_error_response("Modele non charge", "Le modele n'a pas pu etre initialise au demarrage", 500)
    
    try:
        user_id, num_recommendations, exclude_seen = extract_request_params(req)
        if not user_id:
            return create_error_response("user_id manquant", "Veuillez fournir un user_id", 400)
        
        # Génération des recommandations et vérification de l'existence de l'utilisateur
        recommendations, user_exists = generate_recommendations(
            MODEL, user_id, num_recommendations, exclude_seen
        )

        user_known_msg = "Oui" if user_exists else "Non"

        response_data = {
            "ID de l'utilisateur": user_id,
            "L'utilisateur est-il déjà connu ?": user_known_msg,
            "Nombre d'articles recommandés": len(recommendations),
            "Numéros des articles recommandés": recommendations  
        }
        
        return func.HttpResponse(
            json.dumps(response_data, ensure_ascii=False),
            status_code=200,
            headers={'Content-Type': 'application/json; charset=utf-8'}
        )
        
    except Exception as e:
        return create_error_response("Erreur interne du moteur de recommandation", str(e), 500)


# ──────────────── APPEL DU MODELE DE RECOMMANDATION ────────────────
def generate_recommendations(model, user_id: str, num_recommendations: int = 5, exclude_seen: bool = True) -> tuple[list[str], bool]:
    """
    Génère une liste d'articles recommandés pour un utilisateur donné.
    
    Args:
        model: Modèle SVD chargé.
        user_id (str): Identifiant de l'utilisateur.
        num_recommendations (int, optional): Nombre d'articles à recommander. Défaut 5.
        exclude_seen (bool, optional): Exclure les articles déjà vus par l'utilisateur. Défaut True.
    
    Returns:
        tuple[list[str], bool]: 
            - Liste des identifiants d'articles recommandés.
            - Booléen indiquant si l'utilisateur existe dans le trainset.
    """
    trainset = model.trainset
    all_items = list(trainset.all_items())

    # Vérifier si l'utilisateur existe dans le trainset
    try:
        user_inner_id = trainset.to_inner_uid(user_id)
        seen_items = set([trainset.to_raw_iid(inner_id) for (inner_id, _) in trainset.ur[user_inner_id]])
        user_exists = True
    except ValueError:
        seen_items = set()
        user_exists = False

    if user_exists:
        # Filtrer les articles déjà vus si demandé
        if exclude_seen:
            candidate_items = [trainset.to_raw_iid(inner_id) for inner_id in all_items if trainset.to_raw_iid(inner_id) not in seen_items]
        else:
            candidate_items = [trainset.to_raw_iid(inner_id) for inner_id in all_items]

        # Prédire la note estimée pour chaque article candidat
        predictions = [(item_id, model.predict(user_id, item_id).est) for item_id in candidate_items]

        # Trier les articles par score décroissant et sélectionner les meilleurs
        predictions.sort(key=lambda x: x[1], reverse=True)
        top_items = [str(item_id) for item_id, _ in predictions[:num_recommendations]]
    else:
        # Utilisateur inconnu : recommander les articles les plus populaires
        item_popularity = {
            trainset.to_raw_iid(inner_id): len(trainset.ir[inner_id])
            for inner_id in all_items
        }
        most_popular = sorted(item_popularity.items(), key=lambda x: x[1], reverse=True)
        top_items = [str(iid) for iid, _ in most_popular[:num_recommendations]]

    return top_items, user_exists

# ──────────────── EXTRACTION DES PARAMÈTRES ────────────────
def extract_request_params(req: func.HttpRequest) -> tuple[Optional[int], int, bool]:
    """
    Extrait les paramètres nécessaires à la recommandation depuis la requête HTTP.
    
    Args:
        req (func.HttpRequest): Requête HTTP.
    
    Returns:
        tuple[Optional[int], int, bool]: 
            - user_id (int ou str): Identifiant utilisateur.
            - num_recommendations (int): Nombre de recommandations demandées.
            - exclude_seen (bool): Exclure les articles déjà vus.
    """
    user_id = req.params.get('user_id')
    num_recommendations = int(req.params.get('num_recommendations', 10))
    exclude_seen = req.params.get('exclude_seen', 'true').lower() == 'true'

    # Si user_id absent des paramètres, tenter de le récupérer du corps JSON
    if not user_id:
        try:
            req_body = req.get_json()
            if req_body:
                user_id = req_body.get('user_id')
                num_recommendations = req_body.get('num_recommendations', 10)
                exclude_seen = req_body.get('exclude_seen', True)
        except ValueError:
            pass

    # Conversion en int si possible
    try:
        user_id = int(user_id)
    except (TypeError, ValueError):
        pass

    # Limiter le nombre de recommandations entre 1 et 100
    num_recommendations = max(1, min(num_recommendations, 100))
    return user_id, num_recommendations, exclude_seen

# ──────────────── RÉPONSE D'ERREUR ────────────────
def create_error_response(error: str, message: str, status_code: int) -> func.HttpResponse:
    """
    Crée une réponse HTTP standardisée pour les erreurs.
    
    Args:
        error (str): Type d'erreur.
        message (str): Message d'erreur détaillé.
        status_code (int): Code HTTP à retourner.
    
    Returns:
        func.HttpResponse: Réponse HTTP formatée.
    """
    return func.HttpResponse(
        json.dumps({
            "error": error,
            "message": message,
            "status": "error"
        }),
        status_code=status_code,
        headers={'Content-Type': 'application/json'}
    )

# ══════════════════════════════════════════════════════════════════════
# ──────────────── FONCTION UTILITAIRE ET DE DIAGNOSTIC ────────────────
# ══════════════════════════════════════════════════════════════════════
@app.function_name(name="HealthCheck")
@app.route(route="health", methods=["GET"])
def health_check(req: func.HttpRequest) -> func.HttpResponse:
    """
    Vérifie l'état de santé du service de recommandation.
    
    Args:
        req (func.HttpRequest): Requête HTTP.
    
    Returns:
        func.HttpResponse: Réponse HTTP indiquant le statut du service.
    """
    return func.HttpResponse(
        json.dumps({
            "status": "healthy", 
            "message": "Recommendation service is running",
            "architecture": "Azure Function + Blob Storage Input Binding"
        }),
        status_code=200,
        headers={'Content-Type': 'application/json'}
    )
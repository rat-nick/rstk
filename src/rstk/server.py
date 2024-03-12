"""
Module used for serving engines as HTTP endpoints.
"""

from flask import Flask, jsonify, request

from .engine import Engine


# serve the given model on the given port with flask
def serve(model: Engine, port: int):
    """
    Starts the Flask app and exposes the recommend endpoint.

    Args:
        model (Recommender): The recommender model used to generate recommendations.
        port (int): The port number on which the Flask app will run.

    Returns:
        Flask: The Flask app instance.
    """

    app = Flask(__name__)

    @app.route("/", methods=["GET"])
    def recommend():
        """
        Get recommendations based on user preferences, ratings, and profile.

        Parameters:
            None

        Returns:
            A JSON response containing the recommended items.
        """
        preference = request.args.get("preference")
        if preference is not None:
            preference = [x for x in preference.split(",")]

        profile = request.args.get("profile")
        if profile is not None:
            profile = [x for x in profile.split(",")]

        ratings = request.args.get("ratings")
        if ratings is not None:
            ratings = {
                x.split(":")[0]: int(x.split(":")[1]) for x in ratings.split(",")
            }

        k = int(request.args.get("k"))

        return jsonify(model.recommend(profile, ratings, preference, k))

    app.run(port=port)

    return app

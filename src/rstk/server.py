from flask import Flask, jsonify, request

from .recommender import Recommender


# serve the given model on the given port with flask
def serve(model: Recommender, port: int):
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

        preference = request.args.get("preference")
        if preference is not None:
            preference = [int(x) for x in preference.split(",")]

        profile = request.args.get("profile")
        if profile is not None:
            profile = [int(x) for x in profile.split(",")]

        ratings = request.args.get("ratings")
        if ratings is not None:
            ratings = {
                int(x.split(":")[0]): int(x.split(":")[1]) for x in ratings.split(",")
            }

        k = int(request.args.get("k"))

        return jsonify(model.get_recommendations(profile, ratings, preference, k))

    app.run(port=port)

    return app

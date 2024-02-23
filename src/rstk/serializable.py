import json


class Serializable:
    def serialize(self, path: str):
        with open(path, "wb") as f:
            json.dump(self.__dict__, f)

    @staticmethod
    def deserialize(path: str) -> "Serializable":
        with open(path, "rb") as f:
            return json.load(f)

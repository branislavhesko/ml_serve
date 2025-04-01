import abc


class BaseHandler(abc.ABC):
    @abc.abstractmethod
    def predict(self, input: dict[str, bytes]) -> dict[str, bytes]:
        pass

    @abc.abstractmethod
    def get_model_name(self) -> str:
        pass

    @abc.abstractmethod
    def get_model_version(self) -> str:
        pass

    @abc.abstractmethod
    def get_model_path(self) -> str:
        pass

    @abc.abstractmethod
    def get_num_workers(self) -> int:
        pass

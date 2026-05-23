class ParallelMemoryError(Exception):
    base_error_class = True


class MemoryError(ParallelMemoryError):
    def __init__(self, message: str, memory_id: str = None):
        self.memory_id = memory_id
        super().__init__(message)


class EmbeddingError(ParallelMemoryError):
    def __init__(self, message: str, model_name: str = None):
        self.model_name = model_name
        super().__init__(message)


class ModelInferenceError(ParallelMemoryError):
    def __init__(self, message: str, model_type: str = None):
        self.model_type = model_type
        super().__init__(message)


class DatabaseError(ParallelMemoryError):
    def __init__(self, message: str, db_path: str = None):
        self.db_path = db_path
        super().__init__(message)


class FeedbackAggregationError(ParallelMemoryError):
    pass


class GitHubSyncError(ParallelMemoryError):
    def __init__(self, message: str, operation: str = None):
        self.operation = operation
        super().__init__(message)


class UserValidationError(ParallelMemoryError):
    def __init__(self, message: str, user_id: str = None):
        self.user_id = user_id
        super().__init__(message)


class ModelLoadError(ParallelMemoryError):
    def __init__(self, message: str, fallback_available: bool = False):
        self.fallback_available = fallback_available
        super().__init__(message)


class ConfigurationError(ParallelMemoryError):
    pass

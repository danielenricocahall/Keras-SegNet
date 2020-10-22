import tensorflow as tf
from pytest import fixture


@fixture
def tf_session():
    session = tf.Session()
    yield session
    session.close()
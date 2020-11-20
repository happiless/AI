import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
import Generator
from keras.applications.densenet import DenseNet121
from keras.callbacks import TensorBoard
from keras.layers import GlobalAveragePooling2D
from keras.models import Model
from keras.layers import Dense
from keras.optimizers import RMSprop


def train(classes=2, steps_per_epoch=326, validation_steps=67, epochs=50):
    train_generator = Generator.train_data()
    valid_generator = Generator.valid_data()
    visualization = TensorBoard(log_dir='./logs', write_graph=True)
    base_model = DenseNet121(include_top=False)
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(1024, activation='relu')(x)
    predictions = Dense(classes, activation='softmax')(x)
    model = Model(inputs=base_model.input, outputs=predictions)
    # model.summary()
    for layer in base_model.layers:
        layer.trainable = True
    model.compile(optimizer=RMSprop(learning_rate=0.001, decay=0.9, epsilon=0.1),
                  loss='categorical_crossentropy', metrics=['accuracy'])
    model.fit_generator(train_generator,
                        steps_per_epoch=steps_per_epoch,
                        epochs=epochs,
                        validation_data=valid_generator,
                        validation_steps=validation_steps,
                        verbose=1,
                        callbacks=[visualization]
                        )
    model.save('./model/densenet.h5')


train()

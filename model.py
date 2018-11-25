from keras.applications.mobilenetv2 import MobileNetV2
from keras.layers import Dense, Input
from keras.models import Model
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
from sklearn.model_selection import train_test_split, StratifiedKFold
from glob import glob
from utils import *
from sklearn.metrics import roc_auc_score


def build_model(target_size):
    input_tensor = Input(shape=(target_size, target_size, 3))
    base_model = MobileNetV2(
        alpha=0.35,
        include_top=False,
        weights='imagenet',
        input_tensor=input_tensor,
        input_shape=(target_size, target_size, 3),
        pooling='avg')

    for layer in base_model.layers:
        layer.trainable = True

    op = base_model.output

    output_tensor = Dense(1, activation='sigmoid')(op)

    model = Model(inputs=input_tensor, outputs=output_tensor)
    return model


def prepare_data():
    dir1 = 'with_glasses'
    dir2 = 'without_glasses'
    X_with = glob('./dataset/{}/*.jpg'.format(dir1))
    X_without = glob('./dataset/{}/*.jpg'.format(dir2))
    #X_with = X_with[:1000]
    #X_without = X_without[:1000]
    X = np.array(X_with + X_without)
    y = np.array([1] * len(X_with) + [0] * len(X_without))
    return X, y

def prepare_data_kfold(k=5):
    X, y = prepare_data()
    folds = list(StratifiedKFold(n_splits=k, shuffle=True, random_state=1).split(X, y))
    return X, y, folds

def get_callbacks(name_weights):
    mcp_save = ModelCheckpoint(name_weights, save_best_only=True, save_weights_only=True, monitor='val_loss', mode='min')
    return [mcp_save]


def save_model_architecture(model, args):
    model_json = model.to_json()
    with open(args.model_folder + '/' + args.model + '.json', "w") as json_file:
        json_file.write(model_json)

def train(args):
    model = build_model(args.img_size)
    print(model.summary())
    model.compile(optimizer=Adam(),
                  loss='binary_crossentropy',
                  metrics=['binary_accuracy'])
    if args.use_kfold:
        print('Starting...')
        X, y, folds = prepare_data_kfold(k=5)
        for j, (train_idx, val_idx) in enumerate(folds):
            print('Fold ', j)
            X_train_cv = X[train_idx]
            y_train_cv = y[train_idx]
            X_test_cv = X[val_idx]
            y_test_cv = y[val_idx]
            train_generator = load_data_generator(X_train_cv, y_train_cv, batch_size=args.batch_size)
            test_generator = load_data_generator(X_test_cv, y_test_cv, batch_size=args.batch_size)
            callbacks = get_callbacks(name_weights=args.model_folder + '/' +  args.model+'_weigths.h5')
            model.fit_generator(
                generator=train_generator,
                steps_per_epoch=int(X_train_cv.shape[0] / args.batch_size),
                validation_data=test_generator,
                validation_steps=int(X_test_cv.shape[0] / args.batch_size),
                verbose=1,
                epochs=args.epoch,
                workers=10,
                use_multiprocessing=True,
                callbacks=callbacks
            )

            batch_size = 1
            y_pred = model.predict_generator(load_data_test_generator(X_test_cv, batch_size=batch_size),
                                             steps=int(X_test_cv.shape[0] / batch_size))

            print('roc_auc_score:', roc_auc_score(y_test_cv, y_pred))

    else:
        print('Starting...')
        X, y = prepare_data()

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42, stratify=y)

        train_generator = load_data_generator(X_train, y_train, batch_size=args.batch_size)
        test_generator = load_data_generator(X_test, y_test, batch_size=args.batch_size)
        callbacks = get_callbacks(name_weights=args.model_folder + '/' + args.model + '_weigths.h5')
        model.fit_generator(
            generator=train_generator,
            steps_per_epoch=int(X_train.shape[0] / args.batch_size),
            validation_data=test_generator,
            validation_steps=int(X_test.shape[0] / args.batch_size),
            verbose=1,
            epochs=args.epoch,
            workers=10,
            use_multiprocessing=True,
            callbacks=callbacks
        )

        batch_size = 1
        y_pred = model.predict_generator(load_data_test_generator(X_test, batch_size=batch_size),
                                         steps=int(X_test.shape[0] / batch_size))

        print('roc_auc_score:', roc_auc_score(y_test, y_pred))
    save_model_architecture(model, args)

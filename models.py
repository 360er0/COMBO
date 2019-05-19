import tensorflow as tf

from keras import backend as K
from keras import regularizers
from keras.layers import (
    Input,
    GlobalMaxPooling1D,
    TimeDistributed,
    Masking,
    Lambda,
    Bidirectional,
    LSTM,
    Concatenate,
    Conv1D,
    Dense,
    Dot,
    Activation,
    Dropout,
    GaussianNoise,
    RepeatVector,
    GaussianDropout,
)
from keras.layers.embeddings import Embedding
from keras.losses import categorical_crossentropy
from keras.models import Model
from keras.optimizers import Adam


class RemoveMask(Lambda):
    def __init__(self):
        super(RemoveMask, self).__init__((lambda x, mask: x))
        self.supports_masking = True

    def compute_mask(self, input, input_mask=None):
        return None


class KerasModel:

    def __init__(self):
        self.model = self.create()

    def create(self):
        pass

    def __call__(self, input):
        return self.model(input)

    def __getstate__(self):
        state = self.__dict__.copy()
        del state['model']
        state['weights'] = self.model.get_weights()
        return state

    def __setstate__(self, state):
        self.__dict__ = state
        self.model = self.create()
        self.model.set_weights(self.weights)
        del self.weights


class CharModel(KerasModel):

    def __init__(self, params, features_factory, **kwargs):
        self.params = params
        self.features_factory = features_factory
        self.model = self.create()

    def create(self):
        char_embed = Embedding(
            input_dim=self.features_factory.encoders['char'].vocab_size,
            output_dim=self.params.char_embed,
            mask_zero=False,
            weights=None,
            trainable=(not self.params.freeze),
            embeddings_regularizer=regularizers.l2(0.00001),
        )

        conv1 = Conv1D(
            filters=self.params.char_embed*8,
            kernel_size=3,
            strides=1,
            dilation_rate=1,
            activation='relu',
            padding='same',
            kernel_regularizer=regularizers.l2(0.000001),
            bias_regularizer=regularizers.l2(0.000001),
            trainable=(not self.params.freeze),
        )

        conv2 = Conv1D(
            filters=self.params.char_embed*4,
            kernel_size=3,
            strides=1,
            dilation_rate=2,
            activation='relu',
            padding='same',
            kernel_regularizer=regularizers.l2(0.000001),
            bias_regularizer=regularizers.l2(0.000001),
            trainable=(not self.params.freeze),
        )

        conv3 = Conv1D(
            filters=self.params.char_embed,
            kernel_size=3,
            strides=1,
            dilation_rate=4,
            activation=None,
            padding='same',
            kernel_regularizer=regularizers.l2(0.000001),
            bias_regularizer=regularizers.l2(0.000001),
            trainable=(not self.params.freeze),
        )

        single_char_input = Input(shape=(self.params.char_max_len + 2, ))
        char_emb = char_embed(single_char_input)
        char_emb = conv3(conv2(conv1(char_emb)))
        char_emb = GlobalMaxPooling1D()(char_emb)
        char_model = Model(inputs=[single_char_input], outputs=char_emb)

        return char_model


class LemmaModel(KerasModel):

    def __init__(self, params, features_factory, targets_factory, **kwargs):
        self.params = params
        self.features_factory = features_factory
        self.targets_factory = targets_factory
        self.model = self.create()

    def create(self):
        char_embed = Embedding(
            input_dim=self.features_factory.encoders['char'].vocab_size,
            output_dim=self.params.lemma_hidden_size,
            mask_zero=False,
            weights=None,
            trainable=(not self.params.freeze),
            embeddings_regularizer=regularizers.l2(0.00001),
        )

        conv1 = Conv1D(
            filters=self.params.lemma_hidden_size,
            kernel_size=3,
            strides=1,
            dilation_rate=1,
            activation='relu',
            padding='same',
            kernel_regularizer=regularizers.l2(0.000001),
            bias_regularizer=regularizers.l2(0.000001),
            trainable=(not self.params.freeze),
        )

        conv2 = Conv1D(
            filters=self.params.lemma_hidden_size,
            kernel_size=3,
            strides=1,
            dilation_rate=2,
            activation='relu',
            padding='same',
            kernel_regularizer=regularizers.l2(0.000001),
            bias_regularizer=regularizers.l2(0.000001),
            trainable=(not self.params.freeze),
        )

        conv3 = Conv1D(
            filters=self.params.lemma_hidden_size,
            kernel_size=3,
            strides=1,
            dilation_rate=4,
            activation='relu',
            padding='same',
            kernel_regularizer=regularizers.l2(0.000001),
            bias_regularizer=regularizers.l2(0.000001),
            trainable=(not self.params.freeze),
        )

        final_conv = Conv1D(
            filters=self.targets_factory.encoders['lemma'].vocab_size,
            kernel_size=1,
            strides=1,
            dilation_rate=1,
            activation=None,
            padding='same',
            kernel_regularizer=regularizers.l2(0.000001),
            bias_regularizer=regularizers.l2(0.000001),
            trainable=(not self.params.freeze),
        )

        char_size = self.params.char_max_len + 2
        word_emb_size = self.params.lstm_hidden_size*2

        merged_input = Input(shape=(char_size + word_emb_size, ))
        single_char = Lambda(lambda x: x[:, :char_size])(merged_input)
        single_word_emb = Lambda(lambda x: x[:, char_size:])(merged_input)
        single_word_emb = Dropout(self.params.dense_droput)(Dense(32, activation='tanh')(single_word_emb))

        char_emb = char_embed(single_char)
        word_emb = RepeatVector(char_size)(single_word_emb)

        emb = Concatenate()([char_emb, word_emb])
        emb = final_conv(conv3(conv2(conv1(emb))))

        pred = Activation(activation='softmax')(emb)
        lemma_model = Model(inputs=merged_input, outputs=pred)

        return lemma_model


class ParserModel(KerasModel):

    def __init__(self, params, features_factory, targets_factory):
        self.params = params
        self.features_factory = features_factory
        self.targets_factory = targets_factory
        self.model = self.create()

    def cycle_loss(self, y_true, y_pred):
        loss = 0.0
        if self.params.cycle_loss_n == 0:
            return loss

        yn = y_pred[:, 1:, 1:]
        for i in range(self.params.cycle_loss_n):
            loss += K.sum(tf.trace(yn))/self.params.batch_size
            yn = K.batch_dot(yn, y_pred[:, 1:, 1:])

        return loss

    def head_loss(self, y_true, y_pred):
        loss = 0.0
        loss += categorical_crossentropy(y_true, y_pred)
        loss += self.params.cycle_loss_weight*self.cycle_loss(y_true, y_pred)

        return loss

    def lemma_loss(self, y_true, y_pred):
        y_pred = K.clip(y_pred, K.epsilon(), 1 - K.epsilon())
        return -K.mean(K.sum(y_true * K.log(y_pred) + (1 - y_true) * K.log(1 - y_pred), axis=-1)) 

    def feats_loss(self, y_true, y_pred):
        loss = 0.0
        slices = self.targets_factory.encoders['feats'].slices
        for cat, (min_idx, max_idx) in slices.items():
            y_pred_cat = Activation('softmax')(y_pred[:, :, min_idx:max_idx])
            y_true_cat = y_true[:, :, min_idx:max_idx]
            loss += categorical_crossentropy(y_true_cat, y_pred_cat)

        return loss

    def _get_inputs(self):
        raw_inputs = {}
        transformed_inputs = []
        if 'form' in self.params.features:
            input = Input(shape=(None, ))

            if self.params.embed_file is not None:
                word_embed = Embedding(
                    input_dim=self.features_factory.encoders['form'].vocab_size,
                    output_dim=self.features_factory.encoders['form'].emb.embed_size,
                    mask_zero=True,
                    weights=[self.features_factory.encoders['form'].emb.word_vectors],
                    trainable=self.params.train_embed,
                )
            else:
                word_embed = Embedding(
                    input_dim=self.features_factory.encoders['form'].vocab_size,
                    output_dim=self.params.form_embed,
                    mask_zero=True,
                    weights=None,
                    trainable=(not self.params.freeze),
                    embeddings_regularizer=regularizers.l2(0.00001),
                )

            # drop_mask = Lambda(lambda x: K.dropout(K.ones_like(x), self.params.input_droput))(input)
            # drop_input = Lambda(lambda x: K.switch(drop_mask, x, K.ones_like(x)))(input)
            # word_emb = word_embed(drop_input)

            word_emb = word_embed(input)
            word_emb = Dropout(self.params.dense_droput)(Dense(self.params.form_embed, activation='tanh')(word_emb))

            raw_inputs['form'] = input
            transformed_inputs.append(word_emb)

        if 'lemma' in self.params.features:
            input = Input(shape=(None, self.params.char_max_len + 2))
            char_embed = TimeDistributed(
                CharModel(self.params, self.features_factory).model,
            )(Masking()(input))

            raw_inputs['lemma'] = input
            transformed_inputs.append(char_embed)

        if 'upostag' in self.params.features:
            input = Input(shape=(None, ))
            pos_emb = Embedding(
                input_dim=self.features_factory.encoders['upostag'].vocab_size,
                output_dim=self.params.pos_embed,
                mask_zero=True,
                weights=None,
                trainable=(not self.params.freeze),
                embeddings_regularizer=regularizers.l2(0.00001),
            )(input)

            raw_inputs['upostag'] = input
            transformed_inputs.append(pos_emb)

        if 'xpostag' in self.params.features:
            input = Input(shape=(None, ))
            xpos_emb = Embedding(
                input_dim=self.features_factory.encoders['xpostag'].vocab_size,
                output_dim=self.params.xpos_embed,
                mask_zero=True,
                weights=None,
                trainable=(not self.params.freeze),
                embeddings_regularizer=regularizers.l2(0.00001),
            )(input)

            raw_inputs['xpostag'] = input
            transformed_inputs.append(xpos_emb)

        if 'feats' in self.params.features:
            input = Input(shape=(None, self.features_factory.encoders['feats'].vocab_size))
            feat_emb = Dropout(self.params.dense_droput)(Dense(
                self.params.feat_embed, 
                activation='tanh', 
                trainable=(not self.params.freeze),
            )(input))

            raw_inputs['feats'] = input
            transformed_inputs.append(feat_emb)

        if 'char' in self.params.features:
            input = Input(shape=(None, self.params.char_max_len + 2))
            char_embed = TimeDistributed(
                CharModel(self.params, self.features_factory).model,
            )(Masking()(input))

            raw_inputs['char'] = input
            transformed_inputs.append(char_embed)

        if len(transformed_inputs) > 1:
            emb = Concatenate(axis=2)(transformed_inputs)
        else:
            emb = transformed_inputs[0]

        return raw_inputs, emb

    def _get_outputs(self, inputs, emb):

        outputs = {}
        losses = {}
        if 'head' in self.params.targets:
            dep_arc_emb = Dropout(self.params.dense_droput)(Dense(self.params.head_hidden_size, activation='tanh')(emb))
            head_arc_emb = Dropout(self.params.dense_droput)(Dense(self.params.head_hidden_size, activation='tanh')(emb))

            head_pred = Dot(axes=2)([dep_arc_emb, head_arc_emb])
            head_pred = Activation('softmax', name='head')(head_pred)

            outputs['head'] = head_pred
            losses['head'] = self.head_loss

        if 'deprel' in self.params.targets:
            dep_rel_emb = Dropout(self.params.dense_droput)(Dense(self.params.deprel_hidden_size, activation='tanh')(emb))
            head_rel_emb = Dropout(self.params.dense_droput)(Dense(self.params.deprel_hidden_size, activation='tanh')(emb))

            n_deprel = self.targets_factory.encoders['deprel'].vocab_size
            head_emb_T = Lambda(lambda x: K.permute_dimensions(x, (0, 2, 1)))(head_rel_emb)
            deprel_pred = Dot(axes=2)([head_pred, head_emb_T])
            deprel_pred = Concatenate(axis=2)([deprel_pred, dep_rel_emb])
            deprel_pred = Dropout(self.params.dense_droput)(Dense(n_deprel)(deprel_pred))
            deprel_pred = Activation('softmax', name='deprel')(deprel_pred)

            outputs['deprel'] = deprel_pred
            losses['deprel'] = categorical_crossentropy

        if 'lemma' in self.params.targets:

            lemma_pred = TimeDistributed(
                LemmaModel(self.params, self.features_factory, self.targets_factory).model, 
                name='lemma',
            )(Concatenate()([inputs['char'], emb]))

            outputs['lemma'] = lemma_pred
            losses['lemma'] = self.lemma_loss

        if 'xpostag' in self.params.targets:
            n_xpos = self.targets_factory.encoders['xpostag'].vocab_size
            xpos_pred = Dropout(self.params.dense_droput)(Dense(self.params.xpos_hidden_size, activation='tanh')(emb))
            xpos_pred = Dropout(self.params.dense_droput)(Dense(n_xpos)(xpos_pred))
            xpos_pred = Activation('softmax', name='xpostag')(xpos_pred)

            outputs['xpostag'] = xpos_pred
            losses['xpostag'] = categorical_crossentropy

        if 'upostag' in self.params.targets:
            n_pos = self.targets_factory.encoders['upostag'].vocab_size
            pos_pred = Dropout(self.params.dense_droput)(Dense(self.params.pos_hidden_size, activation='tanh')(emb))
            pos_pred = Dropout(self.params.dense_droput)(Dense(n_pos)(pos_pred))
            pos_pred = Activation('softmax', name='upostag')(pos_pred)

            outputs['upostag'] = pos_pred
            losses['upostag'] = categorical_crossentropy

        if 'feats' in self.params.targets:
            n_feat = self.targets_factory.encoders['feats'].vocab_size
            feat_pred = Dropout(self.params.dense_droput)(Dense(self.params.feat_hidden_size, activation='tanh')(emb))
            feat_pred = Dropout(self.params.dense_droput, name='feats')(Dense(n_feat)(feat_pred))

            outputs['feats'] = feat_pred
            losses['feats'] = self.feats_loss

        if 'sent' in self.params.targets:
            sent_pred = RemoveMask()(emb)
            sent_pred = GlobalMaxPooling1D()(sent_pred)

            outputs['sent'] = sent_pred
            losses['sent'] = None

        if 'semrel' in self.params.targets:
            n_semrel = self.targets_factory.encoders['semrel'].vocab_size
            semrel_pred = Dropout(self.params.dense_droput)(Dense(self.params.semrel_hidden_size, activation='tanh')(emb))
            semrel_pred = Dropout(self.params.dense_droput)(Dense(n_semrel)(semrel_pred))
            semrel_pred = Activation('softmax', name='semrel')(semrel_pred)

            outputs['semrel'] = semrel_pred
            losses['semrel'] = categorical_crossentropy

        return outputs, losses

    def create(self):
        # inputs
        inputs, emb = self._get_inputs()

        emb = GaussianDropout(self.params.input_droput)(emb)
        emb = GaussianNoise(0.2)(emb)

        # lstm
        for _ in range(self.params.lstm_layers):
            emb = Bidirectional(
                LSTM(
                    units=self.params.lstm_hidden_size,
                    dropout=self.params.lstm_dropout,
                    recurrent_dropout=self.params.lstm_dropout,
                    return_sequences=True,
                    trainable=(not self.params.freeze),
                    kernel_regularizer=regularizers.l2(0.000001),
                    bias_regularizer=regularizers.l2(0.000001),
                    recurrent_regularizer=regularizers.l2(0.000001),
                    activity_regularizer=regularizers.l2(0.000001),
                ),
            )(emb)

            emb = GaussianDropout(self.params.input_droput)(emb)
            emb = GaussianNoise(0.2)(emb)

        # output
        outputs, losses = self._get_outputs(inputs, emb)

        # model
        model = Model(
            inputs=[inputs[f] for f in self.params.features], 
            outputs=[outputs[t] for t in self.params.targets],
        )
        model.compile(
            loss=[losses[t] for t in self.params.targets],
            loss_weights=self.params.loss_weights,
            optimizer=Adam(lr=self.params.learning_rate, clipvalue=5.0, beta_1=0.9, beta_2=0.9, decay=1e-4),
        )

        return model

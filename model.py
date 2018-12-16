# -*- coding: utf-8 -*-
"""
 @Time    : 2018/12/4 下午4:07
 @FileName: model.py
 @author: 王炳宁
 @contact: wangbingning@sogou-inc.com
"""

from torchUtils import *


class MultiHead(nn.Module):
    def __init__(self, vocab_size, n_embedding, n_hidden, n_layer):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size + 4, embedding_dim=n_embedding)
        self.pos_size = 512
        self.pos_embedding = nn.Embedding(self.pos_size, n_embedding // 4)
        self.n_embedding = n_embedding
        self.projection = nn.Linear(n_embedding + n_embedding // 4, n_embedding)
        self.attention = SelfAttention(n_embedding, n_layer)
        self.att = nn.Linear(n_embedding, 1, bias=False)
        self.output = nn.Linear(n_embedding, 1, bias=False)
        self.mask_output = nn.AdaptiveLogSoftmaxWithLoss(in_features=n_embedding, n_classes=vocab_size + 4,
                                                         cutoffs=[510, 2048, 1200 * 6, 15000],
                                                         div_value=2)
        # self.trans = nn.Linear(n_embedding, vocab_size + 4, bias=False)
        # self.embedding.weight = self.trans.weight
        self.criterion = nn.BCELoss()

    def forward(self, inputs):
        [seq, index, target, label] = inputs
        embedding = self.embedding(seq)
        length = seq.size(1)
        pos = torch.arange(length).cuda()
        pos %= self.pos_size
        pos = pos.expand_as(seq)
        pos_embedding = self.pos_embedding(pos)
        encoder_representations = torch.cat([embedding, pos_embedding], -1)
        encoder_representations = F.leaky_relu(self.projection(encoder_representations), inplace=True)
        encoder_representations = self.attention(encoder_representations)
        value = self.att(encoder_representations)
        score = F.softmax(value, 1)
        output = score.transpose(2, 1).bmm(encoder_representations)

        score = torch.sigmoid(self.output(output)).squeeze()

        if label is None:
            return score

        hidden = encoder_representations.gather(1, index.unsqueeze(2).expand(index.size(0), index.size(1),
                                                                             self.n_embedding))
        # mask_loss = F.cross_entropy(F.log_softmax(self.trans(hidden.contiguous().view(-1, self.n_embedding))), target.contiguous().view(-1))
        mask_loss = self.mask_output(hidden.contiguous().view(-1, self.n_embedding), target.contiguous().view(-1))[1]

        return self.criterion(score, label.half()), mask_loss

    def inference(self, seq):
        embedding = self.embedding(seq)
        length = seq.size(1)
        pos = torch.arange(length).cuda()
        pos %= 512
        pos = pos.expand_as(seq)
        pos_embedding = self.pos_embedding(pos)
        encoder_representations = torch.cat([embedding, pos_embedding], -1)
        encoder_representations = F.leaky_relu(self.projection(encoder_representations), inplace=True)
        encoder_representations = self.attention(encoder_representations)
        return encoder_representations


class AttentionBlock(nn.Module):
    def __init__(self, n_input):
        super().__init__()

        self.conv = nn.Conv2d(in_channels=1, out_channels=2 * n_input, kernel_size=(3, n_input), padding=(1, 0))
        self.q_U = nn.Linear(2 * n_input, n_input)
        self.p_U = nn.Linear(2 * n_input, n_input)
        self.v = nn.Linear(n_input, n_input)
        self.project = nn.Linear(n_input, n_input)

        nn.init.xavier_normal_(self.q_U.weight, gain=0.1)
        nn.init.xavier_normal_(self.p_U.weight, gain=0.1)
        nn.init.xavier_normal_(self.v.weight, gain=0.1)
        nn.init.xavier_normal_(self.project.weight, gain=0.1)

    def get_hidden(self, representations, hidden, linear):
        return F.leaky_relu(linear(torch.cat([hidden, representations], -1)), inplace=True)

    def forward(self, representations):
        c1, c2 = F.relu(self.conv(representations.unsqueeze(1)).squeeze(3).transpose(2, 1), inplace=True).split(
            representations.size(-1), -1)

        s1 = self.get_hidden(representations, c1, self.q_U)
        s2 = self.get_hidden(representations, c2, self.p_U)
        score = F.softmax(self.v(s1).bmm(s2.transpose(2, 1)), 2)

        return representations + score.bmm(F.leaky_relu(self.project(representations), inplace=True))


class SelfAttention(nn.Module):
    def __init__(self, n_hidden, n_layer, n_head=6):
        super().__init__()
        self.n_head = n_head
        self.att = nn.ModuleList()
        for _ in range(n_layer):
            en = AttentionBlock(n_hidden)
            # en = MultiHeadBlock(n_hidden)
            ln = nn.LayerNorm(n_hidden)
            self.att.append(nn.Sequential(en, ln))

    def forward(self, representations):
        for one in self.att:
            representations = one(representations)
        return representations


class Encoder(nn.Module):
    def __init__(self, source_vocab_size, n_embedding, n_hidden, n_layer):
        super().__init__()
        self.embedding = nn.Embedding(source_vocab_size + 4, embedding_dim=n_embedding)
        self.rnn = nn.LSTM(input_size=n_embedding, hidden_size=n_hidden, batch_first=True,
                           bidirectional=True)
        self.projection = nn.Linear(2 * n_hidden, n_embedding)
        self.attention = SelfAttention(n_embedding, n_layer)
        self.att = nn.Linear(n_embedding, 1, bias=False)

    def forward(self, inputs):
        word_embedding = self.embedding(inputs)
        # length = inputs.size(1)
        # pos = torch.arange(length).cuda()
        # pos %= self.pos_size
        # pos = pos.expand_as(inputs)
        # pos_embedding = self.pos_embedding(pos)
        # encoder_representations = torch.cat([word_embedding, pos_embedding], -1)
        encoder_representations, _ = self.rnn(word_embedding)
        encoder_representations = F.leaky_relu(self.projection(encoder_representations), inplace=True)
        encoder_representations = self.attention(encoder_representations)

        value = self.att(encoder_representations)
        score = F.softmax(value, 1)
        output = score.transpose(2, 1).bmm(encoder_representations)

        return encoder_representations, output


class Generator(nn.Module):
    def __init__(self,
                 ch_vocab_size,
                 en_vocab_size,
                 n_embedding,
                 n_hidden,
                 n_layer):
        super().__init__()
        self.vocab_size = en_vocab_size
        self.en_embedding = nn.Embedding(en_vocab_size + 4, embedding_dim=n_embedding)
        self.n_embedding = n_embedding
        self.n_hidden = n_hidden
        self.encoder = Encoder(ch_vocab_size + 4, n_embedding, n_hidden, n_layer)
        self.en_to_de = nn.Linear(n_embedding, 2 * n_hidden)
        self.decoder = nn.LSTM(input_size=n_embedding, hidden_size=n_hidden, batch_first=True)
        self.att_U_q = nn.Linear(n_hidden, n_hidden // 2)
        self.att_U_a = nn.Linear(n_embedding, n_hidden // 2)
        self.att_v = nn.Linear(n_hidden // 2, n_hidden // 2)
        self.project = nn.Linear(n_hidden + n_embedding, n_embedding)
        self.output = nn.AdaptiveLogSoftmaxWithLoss(in_features=n_embedding, n_classes=en_vocab_size + 4,
                                                    cutoffs=[520, 2048, 8503],
                                                    div_value=2)

    def forward(self, inputs):
        [question, answer] = inputs
        if question is None:
            return self.inference(answer)
        question_source = question[:, 0:-1]
        question_target = question[:, 1:]
        question_embedding = self.en_embedding(question_source)
        answer_representations, answer_representation = self.encoder(answer)
        (h0, c0) = torch.tanh(self.en_to_de(answer_representation.transpose(0, 1))).split(self.n_hidden, -1)
        question_representations, _ = self.decoder(question_embedding, (h0.contiguous(), c0.contiguous()))
        answer_s = self.att_v(F.leaky_relu(self.att_U_a(answer_representations), inplace=True))
        question_s = F.leaky_relu(self.att_U_q(question_representations), inplace=True)
        score = F.softmax(question_s.bmm(answer_s.transpose(2, 1)), 2)
        attentive_representations = score.bmm(answer_representations)
        hidden = torch.cat([question_representations, attentive_representations], -1)
        hidden = F.leaky_relu(self.project(hidden), inplace=True)
        mask_loss = self.output(hidden.contiguous().view(-1, self.n_embedding), question_target.contiguous().view(-1))[
            1]
        return mask_loss

    def inference(self, answer):
        answer_representations, answer_representation = self.encoder(answer)
        (h0, c0) = torch.tanh(self.en_to_de(answer_representation.transpose(0, 1))).split(self.n_hidden, -1)

        target = torch.LongTensor([[self.vocab_size]] * answer_representations.size(0)).cuda()
        target_embedding = self.en_embedding(target)

        outputs = []

        for pos in range(70):
            question_representations, (h0, c0) = self.decoder(target_embedding, (h0.contiguous(), c0.contiguous()))
            answer_s = self.att_v(F.leaky_relu(self.att_U_a(answer_representations), inplace=True))
            question_s = F.leaky_relu(self.att_U_q(question_representations), inplace=True)
            score = F.softmax(question_s.bmm(answer_s.transpose(2, 1)), 2)
            attentive_representations = score.bmm(answer_representations)
            hidden = torch.cat([question_representations, attentive_representations], -1)
            hidden = F.leaky_relu(self.project(hidden), inplace=True)
            hidden = hidden.contiguous().view(-1, self.n_embedding)
            prediction = self.output.log_prob(hidden)
            target = torch.argmax(prediction, -1)
            outputs.append(get_tensor_data(target))
            target = target.view(-1, 1)
            target_embedding = self.en_embedding(target)
        return torch.LongTensor(outputs).transpose(0, 1).cuda()


class GeneratorSingle(nn.Module):
    def __init__(self,
                 vocab_size,
                 n_embedding,
                 n_hidden,
                 n_layer):
        super().__init__()
        self.vocab_size = vocab_size
        self.n_embedding = n_embedding
        self.n_hidden = n_hidden
        self.encoder = Encoder(vocab_size, n_embedding, n_hidden, n_layer)
        self.en_to_de = nn.Linear(n_embedding, 2 * n_hidden)
        self.decoder = nn.LSTM(input_size=n_embedding, hidden_size=n_hidden, batch_first=True)
        self.att_U_q = nn.Linear(n_hidden, n_hidden // 2)
        self.att_U_a = nn.Linear(n_embedding, n_hidden // 2)
        self.att_v = nn.Linear(n_hidden // 2, n_hidden // 2)
        self.project = nn.Linear(n_hidden + n_embedding, n_embedding)
        self.output = nn.Linear(n_embedding, vocab_size + 4, bias=False)
        self.output.weight = self.encoder.embedding.weight

    def forward(self, inputs):
        [question, answer] = inputs
        if question is None:
            return self.inference(answer)
        question_source = question[:, 0:-1]
        question_target = question[:, 1:]
        question_embedding = self.encoder.embedding(question_source)
        answer_representations, answer_representation = self.encoder(answer)
        (h0, c0) = torch.tanh(self.en_to_de(answer_representation.transpose(0, 1))).split(self.n_hidden, -1)
        question_representations, _ = self.decoder(question_embedding, (h0.contiguous(), c0.contiguous()))
        answer_s = self.att_v(F.leaky_relu(self.att_U_a(answer_representations), inplace=True))
        question_s = F.leaky_relu(self.att_U_q(question_representations), inplace=True)
        score = F.softmax(question_s.bmm(answer_s.transpose(2, 1)), 2)
        attentive_representations = score.bmm(answer_representations)
        hidden = torch.cat([question_representations, attentive_representations], -1)
        hidden = F.leaky_relu(self.project(hidden), inplace=True)
        logit = self.output(hidden.contiguous().view(-1, self.n_embedding))
        return F.cross_entropy(logit, question_target.contiguous().view(-1))

    def inference(self, answer):
        answer_representations, answer_representation = self.encoder(answer)
        (h0, c0) = torch.tanh(self.en_to_de(answer_representation.transpose(0, 1))).split(self.n_hidden, -1)

        target = torch.LongTensor([[self.vocab_size]] * answer_representations.size(0)).cuda()
        target_embedding = self.encoder.embedding(target)

        outputs = []

        for pos in range(70):
            question_representations, (h0, c0) = self.decoder(target_embedding, (h0.contiguous(), c0.contiguous()))
            answer_s = self.att_v(F.leaky_relu(self.att_U_a(answer_representations), inplace=True))
            question_s = F.leaky_relu(self.att_U_q(question_representations), inplace=True)
            score = F.softmax(question_s.bmm(answer_s.transpose(2, 1)), 2)
            attentive_representations = score.bmm(answer_representations)
            hidden = torch.cat([question_representations, attentive_representations], -1)
            hidden = F.leaky_relu(self.project(hidden), inplace=True)
            hidden = hidden.contiguous().view(-1, self.n_embedding)
            prediction = F.softmax(self.output(hidden), -1)
            target = torch.argmax(prediction, -1)
            outputs.append(get_tensor_data(target))
            target = target.view(-1, 1)
            target_embedding = self.encoder.embedding(target)
        return torch.LongTensor(outputs).transpose(0, 1).cuda()


def get_subsequent_mask(seq):
    """ For masking out the subsequent info. """

    sz_b, len_s = seq.size()
    subsequent_mask = torch.triu(
        torch.ones((len_s, len_s), device=seq.device, dtype=torch.uint8), diagonal=1)
    subsequent_mask = subsequent_mask.unsqueeze(0).expand(sz_b, -1, -1)  # b x ls x ls

    return subsequent_mask


class StackMultiHeadBlock(nn.Module):
    def __init__(self, n_hidden, n_head, n_layer):
        super().__init__()
        self.encoder = nn.ModuleList()
        self.ln = nn.ModuleList()
        for i in range(n_layer):
            self.encoder.append(MultiHeadBlockSelf(n_hidden, n_head))
            self.ln.append(nn.LayerNorm(n_hidden))

    def forward(self, answer_representations, mask):
        for one, ln in zip(self.encoder, self.ln):
            answer_representations = one(answer_representations, mask)
            answer_representations = ln(answer_representations)
        return answer_representations


class MultiHeadBlock(nn.Module):
    def __init__(self, n_hidden, n_head=12):
        super().__init__()
        self.n_hidden = n_hidden
        self.n_head = n_head
        self.att_U_query = nn.Linear(n_hidden, self.n_head * n_hidden // 2)
        self.att_key_value = nn.Linear(n_hidden, 2 * self.n_head * n_hidden // 2)
        self.att_project = nn.Linear(self.n_head * n_hidden // 2, n_hidden)
        nn.init.xavier_normal_(self.att_U_query.weight)
        nn.init.xavier_normal_(self.att_key_value.weight)
        nn.init.xavier_normal_(self.att_project.weight)

    def forward(self, answer_representations, question_representations):
        batch_size = answer_representations.size(0)
        source_size = answer_representations.size(1)
        target_size = question_representations.size(1)
        answer_value, answer_key = F.leaky_relu(self.att_key_value(answer_representations), inplace=True).split(
            self.n_head * self.n_hidden // 2, -1)
        answer_value = answer_value.view(batch_size, source_size, self.n_head, self.n_hidden // 2).transpose(1,
                                                                                                             2).contiguous().view(
            -1, source_size, self.n_hidden // 2)
        answer_key = answer_key.view(batch_size, source_size, self.n_head, self.n_hidden // 2).transpose(1,
                                                                                                         2).contiguous().view(
            -1, source_size, self.n_hidden // 2)

        question_query = F.leaky_relu(self.att_U_query(question_representations), inplace=True)
        question_query = question_query.view(batch_size, target_size, self.n_head, self.n_hidden // 2).transpose(1,
                                                                                                                 2).contiguous().view(
            -1, target_size, self.n_hidden // 2)

        similarities = F.softmax(question_query.bmm(answer_key.transpose(2, 1)), -1)

        value_representation = similarities.bmm(answer_value).view(batch_size, self.n_head, target_size,
                                                                   self.n_hidden // 2).transpose(1,
                                                                                                 2).contiguous().view(
            batch_size, target_size, -1)
        source_attentive_representations = F.leaky_relu(self.att_project(value_representation), inplace=True)

        return source_attentive_representations


class MultiHeadBlockSelf(nn.Module):
    def __init__(self, n_hidden, n_head=12):
        super().__init__()
        self.n_hidden = n_hidden
        self.n_head = n_head
        self.att_query_key_value = nn.Linear(n_hidden, 3 * self.n_head * n_hidden // 2)
        self.att_project = nn.Linear(self.n_head * n_hidden // 2, n_hidden)
        nn.init.xavier_normal_(self.att_query_key_value.weight)
        nn.init.xavier_normal_(self.att_project.weight)

    def forward(self, representations, mask):
        batch_size = representations.size(0)
        source_size = representations.size(1)
        answer_value, answer_key, question_query = F.leaky_relu(self.att_query_key_value(representations),
                                                                inplace=True).split(
            self.n_head * self.n_hidden // 2, -1)
        answer_value = answer_value.view(batch_size, source_size, self.n_head, self.n_hidden // 2).transpose(1,
                                                                                                             2).contiguous().view(
            -1, source_size, self.n_hidden // 2)
        answer_key = answer_key.view(batch_size, source_size, self.n_head, self.n_hidden // 2).transpose(1,
                                                                                                         2).contiguous().view(
            -1, source_size, self.n_hidden // 2)

        question_query = question_query.view(batch_size, source_size, self.n_head, self.n_hidden // 2).transpose(1,
                                                                                                                 2).contiguous().view(
            -1, source_size, self.n_hidden // 2)

        similarities = F.softmax(question_query.bmm(answer_key.transpose(2, 1)).masked_fill(mask, -np.inf), -1)

        value_representation = similarities.bmm(answer_value).view(batch_size, self.n_head, source_size,
                                                                   self.n_hidden // 2).transpose(1,
                                                                                                 2).contiguous().view(
            batch_size, source_size, -1)
        source_attentive_representations = F.leaky_relu(self.att_project(value_representation), inplace=True)

        return source_attentive_representations + representations


class GeneratorSelfAttention(nn.Module):
    def __init__(self,
                 vocab_size,
                 n_embedding,
                 n_hidden,
                 n_layer,
                 n_head=12):
        super().__init__()
        self.n_head = n_head
        self.vocab_size = vocab_size
        self.n_embedding = n_embedding
        self.n_hidden = n_hidden
        self.encoder = Encoder(vocab_size, n_embedding, n_hidden, n_layer)
        self.en_to_de = nn.Linear(n_embedding, 2 * n_hidden)
        self.decoder = nn.LSTM(input_size=n_embedding, hidden_size=n_hidden, batch_first=True)
        self.target_attention = MultiHeadBlock(n_hidden, n_head)
        self.transform = nn.Linear(2 * n_hidden, n_hidden)
        self.self_attention = StackMultiHeadBlock(n_hidden, n_head, n_layer)
        self.project = nn.Linear(n_hidden, n_embedding)
        self.output = nn.Linear(n_embedding, vocab_size + 4, bias=False)
        self.output.weight = self.encoder.embedding.weight

    def forward(self, inputs):
        [question, answer] = inputs
        if question is None:
            return self.inference(answer)
        question_source = question[:, 0:-1]
        question_target = question[:, 1:]
        question_embedding = self.encoder.embedding(question_source)
        answer_representations, answer_representation = self.encoder(answer)
        (h0, c0) = torch.tanh(self.en_to_de(answer_representation.transpose(0, 1))).split(self.n_hidden, -1)
        question_representations, _ = self.decoder(question_embedding, (h0.contiguous(), c0.contiguous()))

        # source to target attention
        source_attentive_representations = self.target_attention(answer_representations, question_representations)

        # self attention

        len_s = question_representations.size(1)
        b_size = question_representations.size(0)
        subsequent_mask = torch.triu(
            torch.ones((len_s, len_s), device=answer.device, dtype=torch.uint8), diagonal=1)
        mask = subsequent_mask.unsqueeze(0).expand(b_size * self.n_head, -1, -1)  # b x ls x ls

        att_representations = F.leaky_relu(
            self.transform(torch.cat([question_representations, source_attentive_representations], -1)), inplace=True)

        hidden = self.self_attention(att_representations, mask)
        hidden = F.leaky_relu(self.project(hidden), inplace=True)
        logit = F.log_softmax(self.output(hidden.contiguous().view(-1, self.n_embedding)), -1)
        return F.cross_entropy(logit, question_target.contiguous().view(-1))

    def inference(self, answer):
        answer_representations, answer_representation = self.encoder(answer)
        (h0, c0) = torch.tanh(self.en_to_de(answer_representation.transpose(0, 1))).split(self.n_hidden, -1)

        target = torch.LongTensor([[self.vocab_size]] * answer_representations.size(0)).cuda()
        target_embedding = self.encoder.embedding(target)
        outputs = []
        b_size = answer_representations.size(0)
        decoder_hidden = None
        for pos in range(70):
            question_representations, (h0, c0) = self.decoder(target_embedding, (h0.contiguous(), c0.contiguous()))
            attentive_representations = self.target_attention(answer_representations, question_representations)

            tmp_hidden = F.leaky_relu(self.transform(torch.cat([question_representations, attentive_representations], -1)), inplace=True)

            decoder_hidden = tmp_hidden if decoder_hidden is None else torch.cat(
                [decoder_hidden, tmp_hidden], 1)

            subsequent_mask = torch.triu(
                torch.ones((pos + 1, pos + 1), device=answer.device, dtype=torch.uint8), diagonal=1)
            mask = subsequent_mask.unsqueeze(0).expand(b_size * self.n_head, -1, -1)  # b x ls x ls

            hidden = self.self_attention(decoder_hidden, mask)[:, -1, :].unsqueeze(1)

            hidden = F.leaky_relu(self.project(hidden), inplace=True)
            hidden = hidden.contiguous().view(-1, self.n_embedding)
            prediction = F.softmax(self.output(hidden), -1)
            target = torch.argmax(prediction, -1)
            outputs.append(get_tensor_data(target))
            target = target.view(-1, 1)
            target_embedding = self.encoder.embedding(target)
        return torch.LongTensor(outputs).transpose(0, 1).cuda()


if __name__ == '__main__':
    model = GeneratorSelfAttention(10000, 256, 256, 5).cuda()
    source = torch.LongTensor(np.random.random_integers(0, 100, size=[64, 27])).cuda()
    target = torch.LongTensor(np.random.random_integers(0, 100, size=[64, 33])).cuda()

    output = model([target, source])

    print(output.size())

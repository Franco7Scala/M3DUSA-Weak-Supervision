import os
import torch

from src.support.utils import get_base_dir


def main():

    edgelists_dir = os.path.join(get_base_dir(), "mumin", "heterodata", "edgelists")

    TR_q = torch.load(os.path.join(edgelists_dir, "tweet_is_quoted_by_reply.pt"))
    RT_q = torch.load(os.path.join(edgelists_dir, "reply_quote_of_tweet.pt"))

    TR_p = torch.load(os.path.join(edgelists_dir, "tweet_is_replied_by_reply.pt"))
    RT_p = torch.load(os.path.join(edgelists_dir, "reply_reply_to_tweet.pt"))

    torch.save(TR_q, os.path.join(edgelists_dir, "reply_quote_of_tweet.pt"))
    torch.save(RT_q, os.path.join(edgelists_dir, "tweet_is_quoted_by_reply.pt"))
    torch.save(TR_p, os.path.join(edgelists_dir, "reply_reply_to_tweet.pt"))
    torch.save(RT_p, os.path.join(edgelists_dir, "tweet_is_replied_by_reply.pt"))


if __name__ == '__main__':
    main()

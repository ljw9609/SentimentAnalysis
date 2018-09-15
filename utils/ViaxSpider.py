import re
import time
import requests
import pymysql
import traceback


class Comment:
    cookie = "SCF=Am2aQ9kMwxp9odAxf3tqK_iqUmGudx3tXeOtYRGPBXv4jJjXNEolHcdN9dKjcp6Cjlo2xqnnqLKgaMJfpw-VfxA.; "
    "_T_WM=d1f77ad29ee1c3298d97bd051a213da8; SUB=_2A253sLWUDeRhGedJ61QU-S7FyjiIHXVVWtvcrDV6PUJ"
    "bkdAKLVfCkW1NVnga0naK_h57OGJDo5JYHQe-j9iDr0uA; SUHB=0ebcw_WHNJb69U; SSOLoginState=1521796548"

    def __init__(self):
        self.header = {
                'User-agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_13_4) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/66.0.3359.139 Safari/537.36',
                'Host': 'm.weibo.cn',
                'Accept': 'application/json, text/plain, */*; q=0.01',
                'Accept-Language': 'zh-CN,zh;q=0.9,en-US;q=0.5,en;q=0.3',
                'Accept-Encoding': 'gzip, deflate, br',
                'DNT': '1',
                'Connection': 'keep-alive',
        }
        self.search_url = 'https://m.weibo.cn/api/container/getIndex?containerid=100103type%3D60%26q%3D{}&page={}'
        self.post_url = 'https://m.weibo.cn/status/{}'
        self.comments_url = 'https://m.weibo.cn/api/comments/show?id={}&page={}'

        self.db = pymysql.connect(host="101.132.180.255", user="root", password="uAiqwVwjJ8-i", db="viax",
                                  port=3306, charset="utf8")

    def request_comments(self, post_id):
        i = 1
        max_page = 100
        while i < max_page:
            r = requests.get(self.comments_url.format(post_id, i), headers=self.header)

            if r.json()['ok'] == 0:
                i += 1
                pass
            else:
                max_page = r.json()['data']['max']
                if max_page > 100:
                    max_page = 100
                data = r.json()['data']['data']
                for comment in data:
                    comment_id = comment['id']
                    user_name = comment['user']['screen_name']
                    user_id = comment['user']['id']
                    profile_url = comment['user']['profile_url']
                    create_at = comment['created_at']
                    text = re.sub('<.*?>|回复<.*?>:|[\U00010000-\U0010ffff]|[\uD800-\uDBFF][\uDC00-\uDFFF]',
                                  '', comment['text'])
                    like_counts = comment['like_counts']
                    source = re.sub('[\U00010000-\U0010ffff]|[\uD800-\uDBFF][\uDC00-\uDFFF]', '',
                                    comment['source'])

                    cur = self.db.cursor()
                    sql = "insert into comments(comment_id, username, create_at, text, source, like_count) values(%s,%s,%s,%s,%s,%s)"
                    param = (comment_id, user_name, create_at, text, source, like_counts)
                    try:
                        cur.execute(sql, param)
                        self.db.commit()
                    except Exception as e:
                        print(e)
                        self.db.rollback()

                i += 1
                time.sleep(5)

        return True

    def run(self):
        page = 1
        try:
            while page < 100:
                r = requests.get(self.search_url.format('加征关税', page), headers=self.header)

                if r.json()['ok'] == 0:
                    page += 1
                    pass
                else:
                    content = r.json()['data']['cards'][0]['card_group']
                    post_ids = [card['mblog']['id'] for card in content]
                    for post_id in post_ids:
                        res = self.request_comments(post_id)
                    page += 1

        except Exception as e:
            print(e)
            print("Current page: " + page)
            traceback.print_exc()


def main():
    comment = Comment()
    comment.run()


if __name__ == "__main__":
    main()



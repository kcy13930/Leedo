from slacker import Slacker
import websocket
import json

token = 'xoxb-1606430616754-1630407758672-lxq48ekcZuHIS4S71PQXnpF5'
slack = Slacker(token)

response = slack.rtm.start()
sock_endpoint = response.body['url']
slack_socket = websocket.create_connection(endpoint)

slack_socket.recv()
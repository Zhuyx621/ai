import sys

from Game import *
from Dot import *
from ChessBoard import *
from ChessAI import *
from MyAI import *
import time
def main():
    
    # 初始化pygame
    pygame.init()
    # 创建用来显示画面的对象（理解为相框）
    screen = pygame.display.set_mode((750, 667))
    # 游戏背景图片
    background_img = pygame.image.load("images/bg.jpg")
    # 游戏棋盘
    # chessboard_img = pygame.image.load("images/bg.png")
    # 创建棋盘对象
    chessboard = ChessBoard(screen)
    # 创建计时器
    clock = pygame.time.Clock()
    # 创建游戏对象（像当前走棋方、游戏是否结束等都封装到这个对象中）
    game = Game(screen, chessboard)
    game.back_button.add_history(chessboard.get_chessboard_str_map())
    # 创建AI对象
    ai = ChessAI(game.user_team)
    myai = myAI(game.computer_team)#你为黑方

    # 主循环
    while True:
        # AI行动
        if not game.show_win and not game.show_draw and game.AI_mode and game.get_player() == ai.team:
            start_time = time.time()
            if game.back_button.is_repeated():
                print("获胜方是",game.get_player())
                game.set_win(game.get_player())
            else:
                # AI预测下一步
                row, col, nxt_row, nxt_col = ai.get_next_step(chessboard)
                # 选择棋子
                ClickBox(screen, row, col)
                # 下棋子
                chessboard.move_chess(nxt_row, nxt_col)
                # 清理「点击对象」
                ClickBox.clean()
                # 检测落子后，是否产生了"将军"功能
                if chessboard.judge_attack_general(game.get_player()):
                    print("将军....")
                    # 检测对方是否可以挽救棋局，如果能挽救，就显示"将军"，否则显示"胜利"
                    if chessboard.judge_win(game.get_player()):
                        print("获胜方是",game.get_player())
                        game.set_win(game.get_player())
                    else:
                        # 如果攻击到对方，则标记显示"将军"效果
                        game.set_attack(True)
                else:
                    if chessboard.judge_win(game.get_player()):
                        print("获胜方是",game.get_player())
                        game.set_win(game.get_player())
                    game.set_attack(False)    
                
                if chessboard.judge_draw():
                    print("和棋...")
                    game.set_draw()
                
                # 落子之后，交换走棋方
                game.back_button.add_history(chessboard.get_chessboard_str_map())
                game.exchange()
            end_time = time.time()
            print("opponent move time: ", round(end_time - start_time, 1))

        elif not game.show_win and not game.show_draw and game.AI_mode and game.get_player() == myai.team:
            if game.back_button.is_repeated():
                print("获胜方是",game.get_player())
                game.set_win(game.get_player())
            else:
                # AI预测下一步
                row, col, nxt_row, nxt_col = myai.get_next_step(chessboard)
                # 选择棋子
                ClickBox(screen, row, col)
                # 下棋子
                chessboard.move_chess(nxt_row, nxt_col)
                # 清理「点击对象」
                ClickBox.clean()
                # 检测落子后，是否产生了"将军"功能
                if chessboard.judge_attack_general(game.get_player()):
                    print("将军....")
                    # 检测对方是否可以挽救棋局，如果能挽救，就显示"将军"，否则显示"胜利"
                    if chessboard.judge_win(game.get_player()):
                        print("获胜方是",game.get_player())
                        game.set_win(game.get_player())
                    else:
                        # 如果攻击到对方，则标记显示"将军"效果
                        game.set_attack(True)
                else:
                    if chessboard.judge_win(game.get_player()):
                        print("获胜方是",game.get_player())
                        game.set_win(game.get_player())
                    game.set_attack(False)    
                
                if chessboard.judge_draw():
                    print("和棋...")
                    game.set_draw()

                # 落子之后，交换走棋方
                game.back_button.add_history(chessboard.get_chessboard_str_map())
                game.exchange()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()  # 退出程序

        # 显示游戏背景
        screen.blit(background_img, (0, 0))
        screen.blit(background_img, (0, 270))
        screen.blit(background_img, (0, 540))

        # 显示棋盘以及棋子
        chessboard.show_chessboard_and_chess()

        # 显示游戏相关信息
        game.show()

        # 显示screen这个相框的内容（此时在这个相框中的内容像照片、文字等会显示出来）
        pygame.display.update()
        clock.tick(60) 
        
if __name__ == '__main__':
    main()

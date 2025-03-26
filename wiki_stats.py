#!/usr/bin/python3

from collections import deque
import os
import sys
import math
import array
import time
import matplotlib.pyplot as plt
import networkx as nx
import statistics
import numpy as np

from matplotlib import rc
rc('font', family='Droid Sans', weight='normal', size=14)

import matplotlib.pyplot as plt

class Timer:
    def __init__(self, runner_name='[Anonymous runner]', autostart = True):
        self._runner = runner_name
        if autostart:
            self.start()

    def start(self):
        self._time_start = time.time()
    
    def stop(self):
        time_end = time.time()
        ops_timing = time_end - self._time_start
        print(f'[Timer] {self._runner} was running for {ops_timing:.4f} seconds')


class WikiGraph:

    def load_from_file(self, filename):
        runner_name = '[Ultimate graph loader]'
        print(f'{runner_name} Загружаю граф из файла: ' + filename)
        timer = Timer(runner_name)
        with open(filename) as f:
            (n, _nlinks) = map(int, f.readline().split())
            print(f'{runner_name} Graph contins {n} pages and {_nlinks} links')
            self._num_of_pages = n
            self._titles = []
            self._sizes = array.array('L', [0]*n)
            self._links = array.array('L', [0]*_nlinks)
            self._redirect = array.array('B', [0]*n)
            self._offset = array.array('L', [0]*(n+1))

            current_link_index = 0
            for i in range(n):
                # Считываем название статьи
                title = f.readline().strip()
                self._titles.append(title)

                # Считываем размер, флаг перенаправления, число исходящих ссылок
                line = f.readline().strip()
                size, redirect_flag, num_links = map(int, line.split())

                self._sizes[i] = size
                self._redirect[i] = redirect_flag
                self._offset[i] = current_link_index

                # Считываем num_links ссылок (номера статей)
                for _ in range(num_links):
                    link_target = int(f.readline().strip())
                    self._links[current_link_index] = link_target
                    current_link_index += 1

            # Последний элемент offset — общее количество записанных ссылок
            self._offset[n] = current_link_index
            
            print(f'{runner_name} Граф загружен')
            timer.stop()


    def get_number_of_links_from(self, _id):
        return self._offset[_id + 1] - self._offset[_id]
    
    def get_links_from(self, _id):
        return self._links[self._offset[_id]:self._offset[_id+1]]

    def get_title(self, _id):
        return self._titles[_id]

    def get_page_size(self, _id):
        return self._sizes[_id]

    def get_id(self, title):
        try:
            return self._titles.index(title)
        except ValueError:
            print(f'[Warning] Node with name {title} not found')
            return -1

    def get_number_of_pages(self):
        return self._num_of_pages

    def is_redirect(self, _id):
        return self._redirect[_id] == 1

    def bfs_path(self, start_title, target_title):
        timer = Timer('[BFS_Search]')
        start_id = self.get_id(start_title)
        target_id = self.get_id(target_title)
        if start_id == -1 or target_id == -1:
            print("[BFS_Search] Incorrect start or end title")
            timer.stop()
            return None

        m = self.get_number_of_pages()
        visited = [False] * m
        pred = [-1] * m
        queue = deque()
        queue.append(start_id)
        visited[start_id] = True

        while queue:
            u = queue.popleft()
            if u == target_id:
                break
            for v in self.get_links_from(u):
                if not visited[v]:
                    visited[v] = True
                    pred[v] = u
                    queue.append(v)

        if not visited[target_id]:
            timer.stop()
            return None

        # Восстанавливаем путь
        path = []
        cur = target_id
        while cur != -1:
            path.append(self.get_title(cur))
            cur = pred[cur]
        path.reverse()
        timer.stop()
        return path

    def visaulize(self):
        if self._num_of_pages > 50:
            print('[Visualizer] Cant show that much nodes, sorry bro')
            return
        
        print('[Visualizer] Visualising your graph...')
        
        G = nx.DiGraph()
        print('[Visualizer] Pages:', self._num_of_pages)
        print('[Visualizer] Links', len(self._links))
        # print(self._links)
        for i in range(self._num_of_pages):
            links = self.get_links_from(i)
            for l in links:
                G.add_edge(self.get_title(i), self.get_title(l), weight=1)
                G.out_degree(self.get_title(i), 2)

        elarge = [(u,v) for (u,v,d) in G.edges(data=True) if d['weight'] >0.5] 

        pos = nx.spring_layout(G) # позиции всех вершин
        nx.draw_networkx_nodes(G, pos, node_size=200)
        nx.draw_networkx_edges(G, pos, edgelist=elarge, width=2)

        nx.draw_networkx_labels(G, pos,font_size=10,font_family='sans-serif')

        plt.axis('off')
        plt.savefig("weighted_graph.png") 
        plt.show() 

    def show_stats(self):
        timer = Timer('[Statistics]')
        print('[Statistics] Measuring graph statistics')
        n = self.get_number_of_pages()
        out_links = [self.get_number_of_links_from(i) for i in range(n)]
        min_out = min(out_links)
        max_out = max(out_links)
        count_min_out = out_links.count(min_out)
        count_max_out = out_links.count(max_out)
        avg_out = statistics.mean(out_links)
        stdev_out = statistics.stdev(out_links) if n > 1 else 0

        # Подсчитываем входящие ссылки (из обычных статей vs из перенаправлений)
        in_links = [0] * n       # ссылки с обычных статей
        in_redirects = [0] * n   # ссылки со статей-перенаправлений
        total_redirect_articles = 0

        for i in range(n):
            if wg.is_redirect(i):
                total_redirect_articles += 1
            for dest in wg.get_links_from(i):
                if wg.is_redirect(i):
                    in_redirects[dest] += 1
                else:
                    in_links[dest] += 1

        # Статистика по in_links
        min_in = min(in_links)
        max_in = max(in_links)
        count_min_in = in_links.count(min_in)
        count_max_in = in_links.count(max_in)
        avg_in = statistics.mean(in_links)
        stdev_in = statistics.stdev(in_links) if n > 1 else 0

        # Статистика по in_redirects
        min_in_redir = min(in_redirects)
        max_in_redir = max(in_redirects)
        count_min_in_redir = in_redirects.count(min_in_redir)
        count_max_in_redir = in_redirects.count(max_in_redir)
        avg_in_redir = statistics.mean(in_redirects)
        stdev_in_redir = statistics.stdev(in_redirects) if n > 1 else 0

        
        # Вывод результатов
        total_redirect_percent = (total_redirect_articles / n * 100) if n else 0
        print('=============STATS START=============')
        print(f"Количество статей с перенаправлением: {total_redirect_articles} ({total_redirect_percent:.2f}%)")
        print("Минимальное количество ссылок из статьи:", min_out)
        print("Количество статей с минимальным количеством ссылок:", count_min_out)
        print("Максимальное количество ссылок из статьи:", max_out)
        print("Количество статей с максимальным количеством ссылок:", count_max_out)
        max_out_id = out_links.index(max_out)
        print("Статья с наибольшим количеством ссылок:", wg.get_title(max_out_id))
        print(f"Среднее количество ссылок в статье: {avg_out:.2f} (ср. откл. {stdev_out:.2f})")

        print("Минимальное количество внешних ссылок на статью:", min_in)
        print("Количество статей с минимальным количеством внешних ссылок:", count_min_in)
        print("Максимальное количество внешних ссылок на статью:", max_in)
        print("Количество статей с максимальным количеством внешних ссылок:", count_max_in)
        max_in_id = in_links.index(max_in)
        print("Статья с наибольшим количеством внешних ссылок:", wg.get_title(max_in_id))
        print(f"Среднее количество внешних ссылок на статью: {avg_in:.2f} (ср. откл. {stdev_in:.2f})")

        print("Минимальное количество внешних перенаправлений на статью:", min_in_redir)
        print("Количество статей с минимальным количеством внешних перенаправлений:", count_min_in_redir)
        print("Максимальное количество внешних перенаправлений на статью:", max_in_redir)
        print("Количество статей с максимальным количеством внешних перенаправлений:", count_max_in_redir)
        max_in_redir_id = in_redirects.index(max_in_redir)
        print("Статья с наибольшим количеством внешних перенаправлений:", wg.get_title(max_in_redir_id))
        print(f"Среднее количество внешних перенаправлений на статью: {avg_in_redir:.2f} (ср. откл. {stdev_in_redir:.2f})")
        print('=============STATS END=============')

        timer.stop()

    def show_stats_np(self):
        timer = Timer('[Statistics_np]')
        n = self.get_number_of_pages()

            # Подсчет исходящих ссылок с помощью NumPy
        out_links = np.array([wg.get_number_of_links_from(i) for i in range(n)])
        min_out = np.min(out_links)
        max_out = np.max(out_links)
        count_min_out = np.count_nonzero(out_links == min_out)
        count_max_out = np.count_nonzero(out_links == max_out)
        avg_out = np.mean(out_links)
        stdev_out = np.std(out_links)

        # Подсчет входящих ссылок
        in_links = [0] * n    # ссылки из обычных статей
        in_redirects = [0] * n   # ссылки из статей-перенаправлений
        total_redirect_articles = 0

        for i in range(n):
            if wg.is_redirect(i):
                total_redirect_articles += 1
            for dest in wg.get_links_from(i):
                if wg.is_redirect(i):
                    in_redirects[dest] += 1
                else:
                    in_links[dest] += 1

        in_links = np.array(in_links)
        in_redirects = np.array(in_redirects)

        min_in = np.min(in_links)
        max_in = np.max(in_links)
        count_min_in = np.count_nonzero(in_links == min_in)
        count_max_in = np.count_nonzero(in_links == max_in)
        avg_in = np.mean(in_links)
        stdev_in = np.std(in_links)

        min_in_redir = np.min(in_redirects)
        max_in_redir = np.max(in_redirects)
        count_min_in_redir = np.count_nonzero(in_redirects == min_in_redir)
        count_max_in_redir = np.count_nonzero(in_redirects == max_in_redir)
        avg_in_redir = np.mean(in_redirects)
        stdev_in_redir = np.std(in_redirects)

        total_redirect_percent = (total_redirect_articles / n * 100) if n else 0
        print('=============STATS START=============')
        print(f"Количество статей с перенаправлением: {total_redirect_articles} ({total_redirect_percent:.2f}%)")
        print("Минимальное количество ссылок из статьи:", min_out)
        print("Количество статей с минимальным количеством ссылок:", count_min_out)
        print("Максимальное количество ссылок из статьи:", max_out)
        print("Количество статей с максимальным количеством ссылок:", count_max_out)
        max_out_id = int(np.where(out_links == max_out)[0][0])
        print("Статья с наибольшим количеством ссылок:", wg.get_title(max_out_id))
        print(f"Среднее количество ссылок в статье: {avg_out:.2f} (ср. откл. {stdev_out:.2f})")

        print("Минимальное количество внешних ссылок на статью:", min_in)
        print("Количество статей с минимальным количеством внешних ссылок:", count_min_in)
        print("Максимальное количество внешних ссылок на статью:", max_in)
        print("Количество статей с максимальным количеством внешних ссылок:", count_max_in)
        max_in_id = int(np.where(in_links == max_in)[0][0])
        print("Статья с наибольшим количеством внешних ссылок:", wg.get_title(max_in_id))
        print(f"Среднее количество внешних ссылок на статью: {avg_in:.2f} (ср. откл. {stdev_in:.2f})")

        print("Минимальное количество внешних перенаправлений на статью:", min_in_redir)
        print("Количество статей с минимальным количеством внешних перенаправлений:", count_min_in_redir)
        print("Максимальное количество внешних перенаправлений на статью:", max_in_redir)
        print("Количество статей с максимальным количеством внешних перенаправлений:", count_max_in_redir)
        max_in_redir_id = int(np.where(in_redirects == max_in_redir)[0][0])
        print("Статья с наибольшим количеством внешних перенаправлений:", wg.get_title(max_in_redir_id))
        print(f"Среднее количество внешних перенаправлений на статью: {avg_in_redir:.2f} (ср. откл. {stdev_in_redir:.2f})")
        print('=============STATS END=============')

        timer.stop()

def print_path(path):
    if path: 
        print(f'=============PATH({path[0]}, {path[-1]}) START=============')
        print(*path, sep='\n')
        print(f'=============PATH({path[0]}, {path[-1]}) END===============')

def measure_exec_time(f, *args):
    start_time = time.time()
    res = f(*args)
    end_time = time.time()
    ops_timing = end_time - start_time
    print(f'[Timer] {f.__name__} was running for {ops_timing:.4f} seconds')
    return


if __name__ == '__main__':
    filepath = ''
    # filepath = 'mini_graph.txt'
    # filepath = 'wiki_small.txt'
    filepath = 'wiki.txt'

    if filepath == '' and len(sys.argv) != 2:
        print('Использование: wiki_stats.py <файл с графом статей>')
        sys.exit(-1)
    
    if len(sys.argv) == 2:
        filepath = sys.argv[1]

    if os.path.isfile(filepath):
        wg = WikiGraph()
        wg.load_from_file(filepath)
        path1 = wg.bfs_path('Python', 'Список_файловых_систем')
        path2 = wg.bfs_path('Python', 'Боль')

        print_path(path1)
        print_path(path2)
        # wg.visaulize()
        # wg.show_stats_np()
        wg.show_stats()
    else:
        print('Файл с графом не найден')
        sys.exit(-1)
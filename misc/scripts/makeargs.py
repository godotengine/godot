
text = """
#define FUNC$numR(m_r,m_func,$argt)\\
	virtual m_r m_func($argtp) { \\
		if (Thread::get_caller_ID()!=server_thread) {\\
			m_r ret;\\
			command_queue.push_and_ret( visual_server, &VisualServer::m_func,$argp,&ret);\\
			return ret;\\
		} else {\\
			return visual_server->m_func($argp);\\
		}\\
	}

#define FUNC$numRC(m_r,m_func,$argt)\\
	virtual m_r m_func($argtp) const { \\
		if (Thread::get_caller_ID()!=server_thread) {\\
			m_r ret;\\
			command_queue.push_and_ret( visual_server, &VisualServer::m_func,$argp,&ret);\\
			return ret;\\
		} else {\\
			return visual_server->m_func($argp);\\
		}\\
	}


#define FUNC$numS(m_func,$argt)\\
	virtual void m_func($argtp) { \\
		if (Thread::get_caller_ID()!=server_thread) {\\
			command_queue.push_and_sync( visual_server, &VisualServer::m_func,$argp);\\
		} else {\\
			visual_server->m_func($argp);\\
		}\\
	}

#define FUNC$numSC(m_func,$argt)\\
	virtual void m_func($argtp) const { \\
		if (Thread::get_caller_ID()!=server_thread) {\\
			command_queue.push_and_sync( visual_server, &VisualServer::m_func,$argp);\\
		} else {\\
			visual_server->m_func($argp);\\
		}\\
	}


#define FUNC$num(m_func,$argt)\\
	virtual void m_func($argtp) { \\
		if (Thread::get_caller_ID()!=server_thread) {\\
			command_queue.push( visual_server, &VisualServer::m_func,$argp);\\
		} else {\\
			visual_server->m_func($argp);\\
		}\\
	}

#define FUNC$numC(m_func,$argt)\\
	virtual void m_func($argtp) const { \\
		if (Thread::get_caller_ID()!=server_thread) {\\
			command_queue.push( visual_server, &VisualServer::m_func,$argp);\\
		} else {\\
			visual_server->m_func($argp);\\
		}\\
	}


"""


for i in range(1, 8):

    tp = ""
    p = ""
    t = ""
    for j in range(i):
        if (j > 0):
            tp += ", "
            p += ", "
            t += ", "
        tp += ("m_arg" + str(j + 1) + " p" + str(j + 1))
        p += ("p" + str(j + 1))
        t += ("m_arg" + str(j + 1))

    t = text.replace("$argtp", tp).replace("$argp", p).replace("$argt", t).replace("$num", str(i))
    print(t)

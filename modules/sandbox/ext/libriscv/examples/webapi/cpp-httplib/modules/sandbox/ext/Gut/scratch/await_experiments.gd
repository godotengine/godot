extends SceneTree


class Awaiter:
    extends Node

    func might_await(should):
        if(should):
            print('awaiting')
            await get_tree().create_timer(.5)
            print('awaited')
        else:
            print('not awaiting')

        # return should

    func call_might_wait(should):
       return await might_await(should)


func _init():
    print('hello world')
    var awaiter = Awaiter.new()
    get_root().add_child(awaiter)
    var ret_val = await awaiter.call_might_wait(true)
    print('call_might_wait returned ', ret_val)

    ret_val = await awaiter.might_await(true)
    print('might_wait returned ', ret_val)


    print('done')
    quit()
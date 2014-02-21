/*
** ADOBE SYSTEMS INCORPORATED
** Copyright 2012 Adobe Systems Incorporated. All Rights Reserved.
**
** NOTICE:  Adobe permits you to use, modify, and distribute this file in
** accordance with the terms of the Adobe license agreement accompanying it.
** If you have received this file from a source other than Adobe, then your use,
** modification, or distribution of it requires the prior written permission of Adobe.
*/

package com.adobe.flascc
{
  import flash.display.DisplayObjectContainer;
  import flash.display.Sprite;
  import flash.display.StageScaleMode;
  import flash.events.Event;
  import flash.net.LocalConnection;
  import flash.net.URLRequest;
  import flash.text.TextField;
  import flash.utils.ByteArray;

  import flash.display.*;
  import flash.display3D.*;
  import flash.events.*;
  import GLS3D.*;

  import com.adobe.flascc.vfs.ISpecialFile;

  /**
  * A basic implementation of a console for FlasCC apps.
  * The PlayerKernel class delegates to this for things like read/write,
  * so that console output can be displayed in a TextField on the Stage.
  */
  public class Console extends Sprite implements ISpecialFile
  {
	private var enableConsole:Boolean = true;
    private var _tf:TextField
    private var inputContainer:DisplayObjectContainer

    /**
    * To Support the preloader case you might want to have the Console
    * act as a child of some other DisplayObjectContainer.
    */
    public function Console(container:DisplayObjectContainer = null)
    {
      CModule.rootSprite = container ? container.root : this

      if(CModule.runningAsWorker()) {
        return;
      }

      if(container) {
        container.addChild(this)
        init(null)
      } else {
        addEventListener(Event.ADDED_TO_STAGE, init)
      }
    }

	protected function context_error(e:Event):void {

		trace("Context error!");
	};


	protected function context_created(e:Event):void {

		var s3d:Stage3D = stage.stage3Ds[0];
		var ctx3d:Context3D = s3d.context3D;
		ctx3d.configureBackBuffer(stage.stageWidth, stage.stageHeight, 2, true /*enableDepthAndStencil*/ );
		trace("Stage3D context: " + ctx3d.driverInfo);

		GLAPI.init(ctx3d, null, stage);
		GLAPI.instance.context.clear(0.0, 0.0, 0.0);
		GLAPI.instance.context.present();

		// change to false to prevent running main in the background
		// when Workers are supported
		const runMainBg:Boolean = false


		try
		{
			// PlayerKernel will delegate read/write requests to the "/dev/tty"
			// file to the object specified with this API.
			CModule.vfs.console = this

			// By default we run "main" on a background worker so that
			// console updates show up in real time. Otherwise "startAsync"
			// causes main to run on the UI worker
			if(runMainBg && CModule.canUseWorkers) // start in bg if we have workers
			  CModule.startBackground(this, new <String>[], new <String>[])
			else
			  CModule.startAsync(this)
		}
		catch(e:*)
		{
			// If main gives any exceptions make sure we get a full stack trace
			// in our console
			consoleWrite(e.toString() + "\n" + e.getStackTrace().toString())
			throw e
		}
	};

    /**
    * All of the real FlasCC init happens in this method,
    * which is either run on startup or once the SWF has
    * been added to the stage.
    */
    protected function init(e:Event):void
    {
      inputContainer = new Sprite()
      addChild(inputContainer)

      addEventListener(Event.ENTER_FRAME, enterFrame)

      stage.frameRate = 60
      stage.scaleMode = StageScaleMode.NO_SCALE

      if(enableConsole) {
        _tf = new TextField
        _tf.multiline = true
        _tf.width = stage.stageWidth
        _tf.height = stage.stageHeight 
        inputContainer.addChild(_tf)
      }

		var s3d:Stage3D = stage.stage3Ds[0];
		s3d.addEventListener(Event.CONTEXT3D_CREATE, context_created);
		s3d.addEventListener(ErrorEvent.ERROR, context_error);
		s3d.requestContext3D(Context3DRenderMode.AUTO);
    }

    /**
    * The callback to call when FlasCC code calls the <code>posix exit()</code> function. Leave null to exit silently.
    * @private
    */
    public var exitHook:Function;

    /**
    * The PlayerKernel implementation will use this function to handle
    * C process exit requests
    */
    public function exit(code:int):Boolean
    {
      // default to unhandled
      return exitHook ? exitHook(code) : false;
    }

    /**
    * The PlayerKernel implementation uses this function to handle
    * C IO write requests to the file "/dev/tty" (for example, output from
    * printf will pass through this function). See the ISpecialFile
    * documentation for more information about the arguments and return value.
    */
    public function write(fd:int, bufPtr:int, nbyte:int, errnoPtr:int):int
    {
      var str:String = CModule.readString(bufPtr, nbyte)
      consoleWrite(str)
      return nbyte
    }

    /**
    * The PlayerKernel implementation uses this function to handle
    * C IO read requests to the file "/dev/tty" (for example, reads from stdin
    * will expect this function to provide the data). See the ISpecialFile
    * documentation for more information about the arguments and return value.
    */
    public function read(fd:int, bufPtr:int, nbyte:int, errnoPtr:int):int
    {
      return 0
    }

    /**
    * The PlayerKernel implementation uses this function to handle
    * C fcntl requests to the file "/dev/tty." 
    * See the ISpecialFile documentation for more information about the
    * arguments and return value.
    */
    public function fcntl(fd:int, com:int, data:int, errnoPtr:int):int
    {
      return 0
    }

    /**
    * The PlayerKernel implementation uses this function to handle
    * C ioctl requests to the file "/dev/tty." 
    * See the ISpecialFile documentation for more information about the
    * arguments and return value.
    */
    public function ioctl(fd:int, com:int, data:int, errnoPtr:int):int
    {
      return CModule.callI(CModule.getPublicSymbol("vglttyioctl"), new <int>[fd, com, data, errnoPtr]);
    }

    /**
    * Helper function that traces to the flashlog text file and also
    * displays output in the on-screen textfield console.
    */
    protected function consoleWrite(s:String):void
    {
      trace(s)
      if(enableConsole) {
        _tf.appendText(s)
        _tf.scrollV = _tf.maxScrollV
      }
    }

    /**
    * The enterFrame callback is run once every frame. UI thunk requests should be handled
    * here by calling <code>CModule.serviceUIRequests()</code> (see CModule ASdocs for more information on the UI thunking functionality).
    */
    protected function enterFrame(e:Event):void
    {
	  CModule.serviceUIRequests();
    }

    /**
    * Provide a way to get the TextField's text.
    */
    public function get consoleText():String
    {
        var txt:String = null;

        if(_tf != null){
            txt = _tf.text;
        }
        
        return txt;
    }
  }
}

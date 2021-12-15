using Microsoft.ML.OnnxRuntime;
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Drawing;
using System.Linq;
using System.Threading;
using System.Windows.Forms;
using VideoOS.Platform;
using VideoOS.Platform.Login;
using VideoOS.Platform.SDK.UI.LoginDialog;

namespace Demo
{
	static class Program
	{
        private static readonly Guid IntegrationId = new Guid("14FA2FED-87D2-44E2-88B6-23A21551BD0D");
        private const string IntegrationName = "Media Live Viewer";
        private const string Version = "1.0";
        private const string ManufacturerName = "Sample Manufacturer";
		private static string modelFilePath = System.IO.Directory.GetCurrentDirectory() + @"/FasterRCNN-10.onnx";
		//private static string modelFilePath = System.IO.Directory.GetCurrentDirectory() + @"/ssd_mobilenet_v1_10.onnx";

		/// <summary>
		/// The main entry point for the application.
		/// </summary>
		[STAThread]
		static void Main()
		{
			Application.EnableVisualStyles();
			Application.SetCompatibleTextRenderingDefault(false);

			VideoOS.Platform.SDK.Environment.Initialize();		// Initialize the standalone Environment
			VideoOS.Platform.SDK.Media.Environment.Initialize();        // Initialize the standalone Environment

            EnvironmentManager.Instance.EnvironmentOptions[EnvironmentOptions.HardwareDecodingMode] = "Auto";
            // EnvironmentManager.Instance.EnvironmentOptions[EnvironmentOptions.HardwareDecodingMode] = "Off";
			// EnvironmentManager.Instance.EnvironmentOptions["ToolkitFork"] = "No";

			EnvironmentManager.Instance.TraceFunctionCalls = true;

			DialogLoginForm loginForm = new DialogLoginForm(SetLoginResult, IntegrationId, IntegrationName, Version, ManufacturerName);
			Application.Run(loginForm);
			if (Connected)
			{
				// Run inference
				//var session = new InferenceSession(modelFilePath);
				int gpuDeviceId = 0; // The GPU device ID to execute on
				var session = new InferenceSession(modelFilePath, SessionOptions.MakeSessionOptionWithCudaProvider(gpuDeviceId));
				
				Application.Run(new MainForm(session));
			}

		}

		private static bool Connected = false;
		private static void SetLoginResult(bool connected)
		{
			Connected = connected;
		}
	}
}

using System;
using System.Diagnostics;
using System.Collections.Generic;
using System.Drawing;
using System.IO;
using System.Threading;
using System.Windows.Forms;
using VideoOS.Platform;
using VideoOS.Platform.Live;
using VideoOS.Platform.UI;
using VideoOS.Platform.Util.AdaptiveStreaming;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
//using SixLabors.ImageSharp;
//using SixLabors.ImageSharp.Formats;
//using SixLabors.ImageSharp.PixelFormats;
using SixLabors.ImageSharp.Processing;
using System.Linq;
using SixLabors.ImageSharp.Drawing.Processing;
using Brushes = System.Drawing.Brushes;
using SixLabors.Fonts;
using Font = System.Drawing.Font;
using FontFamily = System.Drawing.FontFamily;
using SixLabors.ImageSharp;
using Configuration = VideoOS.Platform.Configuration;
using SixLabors.ImageSharp.PixelFormats;
using Microsoft.ML;
using static Microsoft.ML.Transforms.Image.ImageResizingEstimator;

namespace Demo
{
	public partial class MainForm : Form
	{
		#region private fields

		private Item _selectItem1;
		private JPEGLiveSource _jpegLiveSource;

		private int _count = 0;

		#endregion

		#region construction and close

		public MainForm()
		{
			InitializeComponent();
			comboBoxResolution.SelectedIndex = 1;
			comboBoxCompression.SelectedIndex = 0;
            comboBoxStreamSelection.SelectedIndex = 0;
		}

        public MainForm(InferenceSession session)
        {
			InitializeComponent();
			comboBoxResolution.SelectedIndex = 1;
			comboBoxCompression.SelectedIndex = 0;
			comboBoxStreamSelection.SelectedIndex = 0;
			this.session = session;
        }

        private void OnClose(object sender, EventArgs e)
		{
			if (_jpegLiveSource != null)
				_jpegLiveSource.Close();
			Close();
		}
		#endregion


		#region Live Click handling 
		private void OnSelect1Click(object sender, EventArgs e)
		{
			if (_jpegLiveSource != null)
			{
				// Close any current displayed JPEG Live Source
				_jpegLiveSource.LiveContentEvent -= JpegLiveSource1LiveNotificationEvent;
				_jpegLiveSource.LiveStatusEvent -= JpegLiveStatusNotificationEvent;
				_jpegLiveSource.Close();
				_jpegLiveSource = null;
				pictureBox1.Image = new Bitmap(1, 1);
			}
            /*
			checkBoxAspect.Checked = false;
			checkBoxFill.Checked = false;
			checkBoxKeyFramesOnly.Checked = false;
			checkBoxCrop.Checked = false;
            */
			ClearAllFlags();
			ResetSelections();

			ItemPickerForm form = new ItemPickerForm();
			form.KindFilter = Kind.Camera;
			form.AutoAccept = true;
			form.Init(Configuration.Instance.GetItems());

			// Ask user to select a camera
			if (form.ShowDialog() == DialogResult.OK)		
			{
				_selectItem1 = form.SelectedItem;
				buttonSelect1.Text = _selectItem1.Name;

				_jpegLiveSource = new JPEGLiveSource(_selectItem1);
				try
				{
					SetResolution();
					_jpegLiveSource.LiveModeStart = true;
					_jpegLiveSource.SetKeepAspectRatio(checkBoxAspect.Checked, checkBoxFill.Checked);
					checkBoxAspect.Enabled = false;
					checkBoxFill.Enabled = false;
					checkBoxKeyFramesOnly.Enabled = false;
					comboBoxResolution.Enabled = !checkBoxAspect.Checked;	// Only allow resolution change, if filling available space
				    _jpegLiveSource.Width = pictureBox1.Width;
				    _jpegLiveSource.Height = pictureBox1.Height;
					_jpegLiveSource.KeyFramesOnly = checkBoxKeyFramesOnly.Checked;
                    SetStreamType(pictureBox1.Width, pictureBox1.Height);
                    _jpegLiveSource.Init();
					_jpegLiveSource.LiveContentEvent += JpegLiveSource1LiveNotificationEvent;
					_jpegLiveSource.LiveStatusEvent += JpegLiveStatusNotificationEvent;

					labelCount.Text = "0";
					buttonPause.Enabled = true;
                    buttonLift.Enabled = true;
					_count = 0;

				} catch (Exception ex)
				{
					MessageBox.Show("Could not Init:" + ex.Message);
					_jpegLiveSource = null;
				}
			} else
			{
				_selectItem1 = null;
				buttonSelect1.Text = "Select Camera ...";
				labelCount.Text = "0";
				labelSize.Text = "";
				labelResolution.Text = "";
				buttonPause.Enabled = false;
                checkBoxAspect.Enabled = true;
                checkBoxFill.Enabled = true;
                checkBoxKeyFramesOnly.Enabled = true;
                buttonLift.Enabled = false;
            }
        }

	    private bool OnMainThread = false;
        private InferenceSession session;

        /// <summary>
        /// This event is called when JPEG is available or some exception has occurred
        /// </summary>
        /// <param name="sender"></param>
        /// <param name="e"></param>
        void JpegLiveSource1LiveNotificationEvent(object sender, EventArgs e)
		{
			if (this.InvokeRequired)
			{
			    if (OnMainThread)
			    {
                    LiveContentEventArgs args = e as LiveContentEventArgs;
			        if (args != null && args.LiveContent != null)
			        {
                        // UI thread is too busy - discard this frame from display
			            args.LiveContent.Dispose();
			        }
			        return;
			    }
			    OnMainThread = true;
			    // Make sure we execute on the UI thread before updating UI Controls
				BeginInvoke(new EventHandler(JpegLiveSource1LiveNotificationEvent), new[] { sender, e });
			}
			else
			{
				LiveContentEventArgs args = e as LiveContentEventArgs;
				if (args != null)
				{
					if (args.LiveContent != null)		
					{
						// Display the received JPEG
						labelSize.Text = ""+args.LiveContent.Content.Length;

						int width = args.LiveContent.Width;
						int height = args.LiveContent.Height;

						MemoryStream ms = new MemoryStream(args.LiveContent.Content);
						Bitmap newBitmap = new Bitmap(ms);
						labelResolution.Text = "" + width + "x" + height;
						if (pictureBox1.Size.Width != 0 && pictureBox1.Size.Height != 0)
						{
							if (!checkBoxAspect.Checked && (newBitmap.Width != pictureBox1.Width || newBitmap.Height != pictureBox1.Height))
							{
								pictureBox1.Image = new Bitmap(newBitmap, pictureBox1.Size);
							}
							else
							{
								pictureBox1.Image = newBitmap;
							}
						}
						if (args.LiveContent.CroppingDefined)
						{
							labelCropRect.Text = "" + args.LiveContent.CropWidth + "x" + args.LiveContent.CropHeight;
						} else
						{
							labelCropRect.Text = "--";
						}
						textBoxDecodingStatus.Text = args.LiveContent.HardwareDecodingStatus;

						ms.Close();
						ms.Dispose();

						MemoryStream ms1 = new MemoryStream(args.LiveContent.Content);
						inferenceLiveView(ms1);
						ms1.Close();
						ms1.Dispose();

						_count++;
						labelCount.Text = "" + _count;

						args.LiveContent.Dispose();
					} else if (args.Exception != null)
					{
						// Handle any exceptions occurred inside toolkit or on the communication to the VMS

					    Bitmap bitmap = new Bitmap(320, 240);
                        Graphics g = Graphics.FromImage(bitmap);
                        g.FillRectangle(Brushes.Black, 0, 0, bitmap.Width, bitmap.Height);
                        g.DrawString("Connection lost to server ...", new Font(FontFamily.GenericMonospace, 12), Brushes.White, new System.Drawing.PointF(20, pictureBox1.Height/2 - 20));
					    g.Dispose();
                        pictureBox1.Image = new Bitmap(bitmap, pictureBox1.Size);
					    bitmap.Dispose();
					}

				}
                OnMainThread = false;
			}
		}

		/// <summary>
		/// This event is called when a Live status package has been received.
		/// </summary>
		/// <param name="sender"></param>
		/// <param name="e"></param>
		void JpegLiveStatusNotificationEvent(object sender, EventArgs e)
		{
			if (this.InvokeRequired)
			{
				BeginInvoke(new EventHandler(JpegLiveStatusNotificationEvent), new[] { sender, e });
			}
			else
			{
				LiveStatusEventArgs args = e as LiveStatusEventArgs;
				if (args != null)
				{
					if ((args.ChangedStatusFlags & StatusFlags.Motion) != 0)
						checkBoxMotion.Checked = (args.CurrentStatusFlags & StatusFlags.Motion) != 0;

					if ((args.ChangedStatusFlags & StatusFlags.Notification) != 0)
						checkBoxNotification.Checked = (args.CurrentStatusFlags & StatusFlags.Notification) != 0;

					if ((args.ChangedStatusFlags & StatusFlags.CameraConnectionLost) != 0)
						checkBoxOffline.Checked = (args.CurrentStatusFlags & StatusFlags.CameraConnectionLost) != 0;

					if ((args.ChangedStatusFlags & StatusFlags.Recording) != 0)
						checkBoxRec.Checked = (args.CurrentStatusFlags & StatusFlags.Recording) != 0;

					if ((args.ChangedStatusFlags & StatusFlags.LiveFeed) != 0)
						checkBoxLiveFeed.Checked = (args.CurrentStatusFlags & StatusFlags.LiveFeed) != 0;

					if ((args.ChangedStatusFlags & StatusFlags.ClientLiveStopped) != 0)
						checkBoxClientLive.Checked = (args.CurrentStatusFlags & StatusFlags.ClientLiveStopped) != 0;

					if ((args.ChangedStatusFlags & StatusFlags.DatabaseFail) != 0)
						checkBoxDBFail.Checked = (args.CurrentStatusFlags & StatusFlags.DatabaseFail) != 0;

					if ((args.ChangedStatusFlags & StatusFlags.DiskFull) != 0)
						checkBoxDiskFull.Checked = (args.CurrentStatusFlags & StatusFlags.DiskFull) != 0;

					Debug.WriteLine("LiveStatus: motion=" + checkBoxMotion.Checked + ", Notification=" + checkBoxNotification.Checked +
					                ", Offline=" + checkBoxOffline.Checked + ", Recording=" + checkBoxRec.Checked);

					if (checkBoxLiveFeed.Checked==false)
					{
						ClearAllFlags();
					}
				}
			}
		}

		private void ClearAllFlags()
		{
			checkBoxMotion.Checked = false;
			checkBoxNotification.Checked = false;
			//checkBoxOffline.Checked = false;
			checkBoxRec.Checked = false;
			checkBoxClientLive.Checked = false;
			checkBoxDBFail.Checked = false;
			checkBoxDiskFull.Checked = false;
			checkBoxLiveFeed.Checked = false;
		}

		private void ResetSelections()
		{
			comboBoxResolution.SelectedIndex = 1;
			comboBoxCompression.SelectedIndex = 0;
			checkBoxClientLive.Checked = false;
			buttonPause.Enabled = false;
			buttonPause.Text = "Pause";
		}

		private void OnResolutionChanged(object sender, EventArgs e)
		{
			if (_jpegLiveSource != null)
			{
				// _jpegLiveSource.LiveModeStart = false;
				SetResolution();
				// _jpegLiveSource.LiveModeStart = true;
			}
		}

		private void SetResolution()
		{
            int width = 0, height = 0;
			switch (comboBoxResolution.SelectedIndex)
			{
				case 0:
					width = 160;
					height = 120;
					break;
				case 1:
					width = 320;
					height = 240;
					break;
				case 2:
					width = 640;
					height = 480;
					break;
				case 3:
					width = 1024;
					height = 780;
					break;
                case 4:
                    width = 1920;
                    height = 1080;
                    break;
            }
            _jpegLiveSource.Width = width;
            _jpegLiveSource.Height = height;
            _jpegLiveSource.SetWidthHeight();

            SetStreamType(width, height);
        }

        private void SetStreamType(int width, int height)
        {
            if (null == _jpegLiveSource)
                return;

            switch (comboBoxStreamSelection.SelectedIndex)
            {
                case 0:
                    _jpegLiveSource.StreamSelectionParams.StreamSelectionType = StreamSelectionType.DefaultStream;
                    break;

                case 1:
                    _jpegLiveSource.StreamSelectionParams.StreamSelectionType = StreamSelectionType.MaximalResolution;
                    break;

                case 2:
                    _jpegLiveSource.StreamSelectionParams.SetStreamAdaptiveResolution(width, height);
                    _jpegLiveSource.StreamSelectionParams.StreamSelectionType = StreamSelectionType.AdaptiveToResolution;
                    break;
            }
        }

        #endregion

        private void OnClick(object sender, EventArgs e)
		{
			if (_jpegLiveSource != null)
			{
				if (_jpegLiveSource.LiveModeStart)				//TODO review when tool kit return msg is OK.
				{
					_jpegLiveSource.LiveModeStart = false;
					buttonPause.Text = "Start";
				}
				else
				{
					_jpegLiveSource.LiveModeStart = true;
					buttonPause.Text = "Pause";
				}
			}
		}

        private void OnQualityChanged(object sender, EventArgs e)
		{
			if (_jpegLiveSource!=null)
			{
				int newQuality;
                if (Int32.TryParse(comboBoxCompression.SelectedItem.ToString(), out newQuality))
                    _jpegLiveSource.Compression = newQuality;
			}
		}

		private void OnCropChanged(object sender, EventArgs e)
		{
			if (_jpegLiveSource!=null)
			{
				if (checkBoxCrop.Checked)
				{
					_jpegLiveSource.Width = 80;
					_jpegLiveSource.Height = 80;
					_jpegLiveSource.SetWidthHeight();
					_jpegLiveSource.SetCropRectangle(100, 100, 80, 80);
				}
				else
				{
					SetResolution();
					_jpegLiveSource.SetCropRectangle(0.0, 0.0, 1.0, 1.0);
				}
			}
		}

		private void OnResizePictureBox(object sender, EventArgs e)
		{
		    if (_jpegLiveSource != null)
		    {
		        _jpegLiveSource.Width = pictureBox1.Width;
		        _jpegLiveSource.Height = pictureBox1.Height;
		        _jpegLiveSource.SetWidthHeight();
		    }

            SetStreamType(pictureBox1.Width, pictureBox1.Height);
        }

        private void comboBoxfps_SelectedIndexChanged(object sender, EventArgs e)
        {
            if (_jpegLiveSource != null)
            {
                var value = comboBoxfps.SelectedItem.ToString();
                if (value.Equals("default", StringComparison.InvariantCultureIgnoreCase))
                {
                    _jpegLiveSource.FPS = 0;
                }
                else
                {
                    int newFps;
                    if (Int32.TryParse(comboBoxfps.SelectedItem.ToString(), out newFps))
                        _jpegLiveSource.FPS = newFps;
                }
            }
        }

        private void buttonLift_Click(object sender, EventArgs e)
        {
            Configuration.Instance.ServerFQID.ServerId.UserContext.SetPrivacyMaskLifted(!Configuration.Instance.ServerFQID.ServerId.UserContext.PrivacyMaskLifted);
        }

        private void comboBoxStreamSelection_SelectedIndexChanged(object sender, EventArgs e)
        {
            SetStreamType(pictureBox1.Width, pictureBox1.Height);
        }

		private void inferenceLiveView(MemoryStream ms)
        {
			ms.Seek(0, SeekOrigin.Begin);
			SixLabors.ImageSharp.Image<SixLabors.ImageSharp.PixelFormats.Rgb24> image = SixLabors.ImageSharp.Image.Load<SixLabors.ImageSharp.PixelFormats.Rgb24>(ms);

			// Resize image
			float ratio = 400f / Math.Min(image.Width, image.Height);
			image.Mutate(x => x.Resize((int)(ratio * image.Width), (int)(ratio * image.Height)));

			// Preprocess image
			var paddedHeight = (int)(Math.Ceiling(image.Height / 32f) * 32f);
			var paddedWidth = (int)(Math.Ceiling(image.Width / 32f) * 32f);
			Tensor<float> input = new DenseTensor<float>(new[] { 3, paddedHeight, paddedWidth });
			var mean = new[] { 102.9801f, 115.9465f, 122.7717f };
			for (int y = paddedHeight - image.Height; y < image.Height; y++)
			{
				Span<SixLabors.ImageSharp.PixelFormats.Rgb24> pixelSpan = image.GetPixelRowSpan(y);
				for (int x = paddedWidth - image.Width; x < image.Width; x++)
				{
					input[0, y, x] = pixelSpan[x].B - mean[0];
					input[1, y, x] = pixelSpan[x].G - mean[1];
					input[2, y, x] = pixelSpan[x].R - mean[2];
				}
			}

			// Setup inputs and outputs
			var inputs = new List<NamedOnnxValue>
			{
				NamedOnnxValue.CreateFromTensor("image", input)
			};
			DateTime startTime = DateTime.Now;
			// Run inference
			//var session = new InferenceSession(modelFilePath);
			IDisposableReadOnlyCollection<DisposableNamedOnnxValue> results = this.session.Run(inputs);

			DateTime endTime = DateTime.Now;
			TimeSpan interval = endTime - startTime;

			Console.Out.WriteLine("   {0,-35} {1,20:N0}", "Total Number of Milliseconds:", interval.TotalMilliseconds);
			// Postprocess to get predictions
			var resultsArray = results.ToArray();
			float[] boxes = resultsArray[0].AsEnumerable<float>().ToArray();
			long[] labels = resultsArray[1].AsEnumerable<long>().ToArray();
			float[] confidences = resultsArray[2].AsEnumerable<float>().ToArray();
			string[] stringArray = { "person", "bicycle", "car", "motorcycle" };
			var predictions = new List<Prediction>();
			var minConfidence = 0.7f;
			for (int i = 0; i < boxes.Length - 4; i += 4)
			{
				var index = i / 4;
				if (confidences[index] >= minConfidence && Array.IndexOf(stringArray, LabelMap.Labels[labels[index]]) > -1 )
				{
					predictions.Add(new Prediction
					{
						Box = new Box(boxes[i], boxes[i + 1], boxes[i + 2], boxes[i + 3]),
						Label = LabelMap.Labels[labels[index]],
						Confidence = confidences[index]
					});
				}
			}

			// Put boxes, labels and confidence on image and save for viewing
			//var outputImage = File.OpenWrite(outImageFilePath);
			SixLabors.Fonts.Font font = SixLabors.Fonts.SystemFonts.CreateFont("Arial", 16);
			foreach (var p in predictions)
			{
				//Console.Out.WriteLine(p.Box.Xmin);
				image.Mutate(x =>
				{
					x.DrawLines(SixLabors.ImageSharp.Color.Red, 2f, new SixLabors.ImageSharp.PointF[] {

						new SixLabors.ImageSharp.PointF(p.Box.Xmin, p.Box.Ymin),
						new SixLabors.ImageSharp.PointF(p.Box.Xmax, p.Box.Ymin),

						new SixLabors.ImageSharp.PointF(p.Box.Xmax, p.Box.Ymin),
						new SixLabors.ImageSharp.PointF(p.Box.Xmax, p.Box.Ymax),

						new SixLabors.ImageSharp.PointF(p.Box.Xmax, p.Box.Ymax),
						new SixLabors.ImageSharp.PointF(p.Box.Xmin, p.Box.Ymax),

						new SixLabors.ImageSharp.PointF(p.Box.Xmin, p.Box.Ymax),
						new SixLabors.ImageSharp.PointF(p.Box.Xmin, p.Box.Ymin)
					});
					x.DrawText($"{p.Label}, {p.Confidence:0.00}", font, SixLabors.ImageSharp.Color.White, new SixLabors.ImageSharp.PointF(p.Box.Xmin, p.Box.Ymin));
				});
			}
			// render onto an Image
			var stream = new System.IO.MemoryStream();
			image.SaveAsBmp(stream);
			System.Drawing.Image img = System.Drawing.Image.FromStream(stream);

			// dispose the old image before displaying the new one
			pictureBox2.Image?.Dispose();
			pictureBox2.Image = new Bitmap(img, pictureBox2.Size);

		}

		private void button1_Click(object sender, EventArgs e)
		{
			string modelFilePath = Directory.GetCurrentDirectory() + @"/yolov4_1_3_416_416_static.onnx";
			string imageFilePath = Directory.GetCurrentDirectory() + @"/demo.jpg";
			string outImageFilePath = Directory.GetCurrentDirectory() + @"/demo_out.jpg";
			int gpuDeviceId = 0; // The GPU device ID to execute on
			var session1 = new InferenceSession(modelFilePath, SessionOptions.MakeSessionOptionWithCudaProvider(gpuDeviceId));

			DateTime startTime = DateTime.Now;
			// Read image
			SixLabors.ImageSharp.Image<SixLabors.ImageSharp.PixelFormats.Rgb24> imageOrg = SixLabors.ImageSharp.Image.Load<SixLabors.ImageSharp.PixelFormats.Rgb24>(imageFilePath);
			//MemoryStream ms = new MemoryStream(args.LiveContent.Content);
			//Bitmap newBitmap = new Bitmap(ms);

			//Letterbox image
			var iw = imageOrg.Width;
			var ih = imageOrg.Height;
			var w = 416;
			var h = 416;

			if ((iw == 0) || (ih == 0))
			{
				Console.WriteLine("Math error: Attempted to divide by Zero");
				return;
			}

			float width = (float)w / iw;
			float height = (float)h / ih;

			float scale = Math.Min(width, height);

			var nw = (int)(iw * scale);
			var nh = (int)(ih * scale);

			var pad_dims_w = (w - nw) / 2;
			var pad_dims_h = (h - nh) / 2;

			// Resize image using default bicubic sampler 
			var image = imageOrg.Clone(x => x.Resize((nw), (nh)));

			var clone = new Image<Rgb24>(w, h);
			clone.Mutate(i => i.Fill(SixLabors.ImageSharp.Color.Gray));
			clone.Mutate(o => o.DrawImage(image, new SixLabors.ImageSharp.Point(pad_dims_w, pad_dims_h), 1f)); // draw the first one top left

			//Preprocessing image
			Tensor<float> input = new DenseTensor<float>(new[] { 1, 3, h, w });
			for (int y = 0; y < clone.Height; y++)
			{
				Span<Rgb24> pixelSpan = clone.GetPixelRowSpan(y);
				for (int x = 0; x < clone.Width; x++)
				{
					input[0, 0, y, x] = pixelSpan[x].B / 255f;
					input[0, 1, y, x] = pixelSpan[x].G / 255f;
					input[0, 2, y, x] = pixelSpan[x].R / 255f;
				}
			}

			//Get the Image Shape
			var image_shape = new DenseTensor<float>(new[] { 1, 2 });
			image_shape[0, 0] = ih;
			image_shape[0, 1] = iw;			

			// Setup inputs and outputs
			var inputs = new List<NamedOnnxValue>
			{
				NamedOnnxValue.CreateFromTensor("input", input)
			};

			// Run inference
			//var session = new InferenceSession(modelFilePath);
			IDisposableReadOnlyCollection<DisposableNamedOnnxValue> results = session1.Run(inputs);
			DateTime endTime = DateTime.Now;
			TimeSpan interval = endTime - startTime;
			
			Console.Out.WriteLine("   {0,-35} {1,20:N0}", "Total Number of Milliseconds:", interval.TotalMilliseconds);

			//Post Processing Steps
			var resultsArray = results.ToArray();
			Tensor<float> boxes = resultsArray[0].AsTensor<float>();
			Tensor<float> scores = resultsArray[1].AsTensor<float>();
			int[] indices = resultsArray[2].AsTensor<int>().ToArray();

			var len = indices.Length / 3;
			var out_classes = new int[len];
			float[] out_scores = new float[len];

			var predictions = new List<Prediction>();
			var count = 0;
			for (int i = 0; i < indices.Length; i = i + 3)
			{
				out_classes[count] = indices[i + 1];
				out_scores[count] = scores[indices[i], indices[i + 1], indices[i + 2]];
				predictions.Add(new Prediction
				{
					Box = new Box(boxes[indices[i], indices[i + 2], 1],
									 boxes[indices[i], indices[i + 2], 0],
									 boxes[indices[i], indices[i + 2], 3],
									 boxes[indices[i], indices[i + 2], 2]),
					Label = LabelMap.Labels[out_classes[count]],
					Confidence = out_scores[count]
				});
				count++;
			}

			// Postprocess to get predictions
			//var resultsArray = results.ToArray();
			//float[] boxes = resultsArray[0].AsEnumerable<float>().ToArray();
			//long[] labels = resultsArray[1].AsEnumerable<long>().ToArray();
			//float[] confidences = resultsArray[2].AsEnumerable<float>().ToArray();
			//var predictions = new List<Prediction>();
			//var minConfidence = 0.7f;
			//for (int i = 0; i < boxes.Length - 4; i += 4)
			//{
			//	var index = i / 4;
			//	if (confidences[index] >= minConfidence)
			//	{
			//		predictions.Add(new Prediction
			//		{
			//			Box = new Box(boxes[i], boxes[i + 1], boxes[i + 2], boxes[i + 3]),
			//			Label = LabelMap.Labels[labels[index]],
			//			Confidence = confidences[index]
			//		});
			//	}
			//}

			// Put boxes, labels and confidence on image and save for viewing
			var outputImage = File.OpenWrite(outImageFilePath);
			SixLabors.Fonts.Font font = SixLabors.Fonts.SystemFonts.CreateFont("Arial", 16);
			foreach (var p in predictions)
			{
				//Console.Out.WriteLine(p.Box.Xmin);
				image.Mutate(x =>
				{
					x.DrawLines(SixLabors.ImageSharp.Color.Red, 2f, new SixLabors.ImageSharp.PointF[] {

						new SixLabors.ImageSharp.PointF(p.Box.Xmin, p.Box.Ymin),
						new SixLabors.ImageSharp.PointF(p.Box.Xmax, p.Box.Ymin),

						new SixLabors.ImageSharp.PointF(p.Box.Xmax, p.Box.Ymin),
						new SixLabors.ImageSharp.PointF(p.Box.Xmax, p.Box.Ymax),

						new SixLabors.ImageSharp.PointF(p.Box.Xmax, p.Box.Ymax),
						new SixLabors.ImageSharp.PointF(p.Box.Xmin, p.Box.Ymax),

						new SixLabors.ImageSharp.PointF(p.Box.Xmin, p.Box.Ymax),
						new SixLabors.ImageSharp.PointF(p.Box.Xmin, p.Box.Ymin)
					});
					x.DrawText($"{p.Label}, {p.Confidence:0.00}", font, SixLabors.ImageSharp.Color.White, new SixLabors.ImageSharp.PointF(p.Box.Xmin, p.Box.Ymin));
				});
			}
			// render onto an Image
			var stream = new System.IO.MemoryStream();
			image.SaveAsBmp(stream);
			System.Drawing.Image img = System.Drawing.Image.FromStream(stream);

			// dispose the old image before displaying the new one
			pictureBox2.Image?.Dispose();
			pictureBox2.Image = new Bitmap(img, pictureBox2.Size);
			//image.SaveAsJpeg(outputImage);
		}

        private void button2_Click(object sender, EventArgs e)
        {
			MLContext mlContext = new MLContext();
			string modelPath = Directory.GetCurrentDirectory() + @"/yolov4.onnx";
			string imageFilePath = Directory.GetCurrentDirectory() + @"/demo.jpg";
			string outImageFilePath = Directory.GetCurrentDirectory() + @"/demo_out.jpg";
			string[] classesNames = new string[] { "person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed", "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush" };

			// Define scoring pipeline
			var pipeline = mlContext.Transforms.ResizeImages(inputColumnName: "bitmap", outputColumnName: "input_1:0", imageWidth: 416, imageHeight: 416, resizing: ResizingKind.IsoPad)
				.Append(mlContext.Transforms.ExtractPixels(outputColumnName: "input_1:0", scaleImage: 1f / 255f, interleavePixelColors: true))
				.Append(mlContext.Transforms.ApplyOnnxModel(
					shapeDictionary: new Dictionary<string, int[]>()
					{
						{ "input_1:0", new[] { 1, 416, 416, 3 } },
						{ "Identity:0", new[] { 1, 52, 52, 3, 85 } },
						{ "Identity_1:0", new[] { 1, 26, 26, 3, 85 } },
						{ "Identity_2:0", new[] { 1, 13, 13, 3, 85 } },
					},
					inputColumnNames: new[]
					{
						"input_1:0"
					},
					outputColumnNames: new[]
					{
						"Identity:0",
						"Identity_1:0",
						"Identity_2:0"
					},
					modelFile: modelPath, 
					recursionLimit: 100, 
					gpuDeviceId: 0));

			// Fit on empty list to obtain input data schema
			var model = pipeline.Fit(mlContext.Data.LoadFromEnumerable(new List<YoloV4BitmapData>()));

			// Create prediction engine
			var predictionEngine = mlContext.Model.CreatePredictionEngine<YoloV4BitmapData, YoloV4Prediction>(model);

			// save model
			//mlContext.Model.Save(model, predictionEngine.OutputSchema, Path.ChangeExtension(modelPath, "zip"));
			var sw = new Stopwatch();
			sw.Start();
			foreach (string imageName in new string[] { "kite.jpg", "dog_cat.jpg", "cars road.jpg", "ski.jpg", "ski2.jpg" })
			{
				using (var bitmap = new Bitmap(System.Drawing.Image.FromFile(imageFilePath)))
				{
					sw.Restart();
					// predict
					var predict = predictionEngine.Predict(new YoloV4BitmapData() { Image = bitmap });
					var results = predict.GetResults(classesNames, 0.3f, 0.7f);

					sw.Stop();
					
					Console.WriteLine($"Done in {sw.ElapsedMilliseconds}ms.");
					
					using (var g = Graphics.FromImage(bitmap))
					{
						foreach (var res in results)
						{
							// draw predictions
							var x1 = res.BBox[0];
							var y1 = res.BBox[1];
							var x2 = res.BBox[2];
							var y2 = res.BBox[3];
							g.DrawRectangle(System.Drawing.Pens.Red, x1, y1, x2 - x1, y2 - y1);
							using (var brushes = new System.Drawing.SolidBrush(System.Drawing.Color.FromArgb(50, System.Drawing.Color.Red)))
							{
								g.FillRectangle(brushes, x1, y1, x2 - x1, y2 - y1);
							}

							g.DrawString(res.Label + " " + res.Confidence.ToString("0.00"),
										 new Font("Arial", 12), Brushes.Blue, new System.Drawing.PointF(x1, y1));
						}
						bitmap.Save(outImageFilePath);
						// dispose the old image before displaying the new one
						pictureBox2.Image?.Dispose();
						pictureBox2.Image = new Bitmap(bitmap, pictureBox2.Size);
					}
				}
				
			}
			
		}

        //private void button1_Click(object sender, EventArgs e)
        //{
        //	string modelFilePath = Directory.GetCurrentDirectory() + @"/FasterRCNN-10.onnx";
        //	string imageFilePath = Directory.GetCurrentDirectory() + @"/demo.jpg";
        //	string outImageFilePath = Directory.GetCurrentDirectory() + @"/demo_out.jpg";

        //	DateTime startTime = DateTime.Now;
        //	// Read image
        //	SixLabors.ImageSharp.Image<SixLabors.ImageSharp.PixelFormats.Rgb24> image = SixLabors.ImageSharp.Image.Load<SixLabors.ImageSharp.PixelFormats.Rgb24>(imageFilePath);
        //	//MemoryStream ms = new MemoryStream(args.LiveContent.Content);
        //	//Bitmap newBitmap = new Bitmap(ms);

        //	// Resize image
        //	float ratio = 800f / Math.Min(image.Width, image.Height);
        //	image.Mutate(x => x.Resize((int)(ratio * image.Width), (int)(ratio * image.Height)));

        //	// Preprocess image
        //	var paddedHeight = (int)(Math.Ceiling(image.Height / 32f) * 32f);
        //	var paddedWidth = (int)(Math.Ceiling(image.Width / 32f) * 32f);
        //	Tensor<float> input = new DenseTensor<float>(new[] { 3, paddedHeight, paddedWidth });
        //	var mean = new[] { 102.9801f, 115.9465f, 122.7717f };
        //	for (int y = paddedHeight - image.Height; y < image.Height; y++)
        //	{
        //		Span<SixLabors.ImageSharp.PixelFormats.Rgb24> pixelSpan = image.GetPixelRowSpan(y);
        //		for (int x = paddedWidth - image.Width; x < image.Width; x++)
        //		{
        //			input[0, y, x] = pixelSpan[x].B - mean[0];
        //			input[1, y, x] = pixelSpan[x].G - mean[1];
        //			input[2, y, x] = pixelSpan[x].R - mean[2];
        //		}
        //	}

        //	// Setup inputs and outputs
        //	var inputs = new List<NamedOnnxValue>
        //	{
        //		NamedOnnxValue.CreateFromTensor("image", input)
        //	};

        //	// Run inference
        //	//var session = new InferenceSession(modelFilePath);
        //	IDisposableReadOnlyCollection<DisposableNamedOnnxValue> results = this.session.Run(inputs);
        //	DateTime endTime = DateTime.Now;
        //	TimeSpan interval = endTime - startTime;

        //	Console.Out.WriteLine("   {0,-35} {1,20:N0}", "Total Number of Milliseconds:", interval.TotalMilliseconds);

        //	// Postprocess to get predictions
        //	var resultsArray = results.ToArray();
        //	float[] boxes = resultsArray[0].AsEnumerable<float>().ToArray();
        //	long[] labels = resultsArray[1].AsEnumerable<long>().ToArray();
        //	float[] confidences = resultsArray[2].AsEnumerable<float>().ToArray();
        //	var predictions = new List<Prediction>();
        //	var minConfidence = 0.7f;
        //	for (int i = 0; i < boxes.Length - 4; i += 4)
        //	{
        //		var index = i / 4;
        //		if (confidences[index] >= minConfidence)
        //		{
        //			predictions.Add(new Prediction
        //			{
        //				Box = new Box(boxes[i], boxes[i + 1], boxes[i + 2], boxes[i + 3]),
        //				Label = LabelMap.Labels[labels[index]],
        //				Confidence = confidences[index]
        //			});
        //		}
        //	}

        //	// Put boxes, labels and confidence on image and save for viewing
        //	var outputImage = File.OpenWrite(outImageFilePath);
        //	SixLabors.Fonts.Font font = SixLabors.Fonts.SystemFonts.CreateFont("Arial", 16);
        //	foreach (var p in predictions)
        //	{
        //		//Console.Out.WriteLine(p.Box.Xmin);
        //		image.Mutate(x =>
        //		{
        //			x.DrawLines(SixLabors.ImageSharp.Color.Red, 2f, new SixLabors.ImageSharp.PointF[] {

        //				new SixLabors.ImageSharp.PointF(p.Box.Xmin, p.Box.Ymin),
        //				new SixLabors.ImageSharp.PointF(p.Box.Xmax, p.Box.Ymin),

        //				new SixLabors.ImageSharp.PointF(p.Box.Xmax, p.Box.Ymin),
        //				new SixLabors.ImageSharp.PointF(p.Box.Xmax, p.Box.Ymax),

        //				new SixLabors.ImageSharp.PointF(p.Box.Xmax, p.Box.Ymax),
        //				new SixLabors.ImageSharp.PointF(p.Box.Xmin, p.Box.Ymax),

        //				new SixLabors.ImageSharp.PointF(p.Box.Xmin, p.Box.Ymax),
        //				new SixLabors.ImageSharp.PointF(p.Box.Xmin, p.Box.Ymin)
        //			});
        //			x.DrawText($"{p.Label}, {p.Confidence:0.00}", font, SixLabors.ImageSharp.Color.White, new SixLabors.ImageSharp.PointF(p.Box.Xmin, p.Box.Ymin));
        //		});
        //	}
        //	// render onto an Image
        //	var stream = new System.IO.MemoryStream();
        //	image.SaveAsBmp(stream);
        //	System.Drawing.Image img = System.Drawing.Image.FromStream(stream);

        //	// dispose the old image before displaying the new one
        //	pictureBox2.Image?.Dispose();
        //	pictureBox2.Image = new Bitmap(img, pictureBox2.Size);
        //	//image.SaveAsJpeg(outputImage);
        //}
    }
}

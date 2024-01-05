//////////////////////////////////////////////////////////////////
// this sketch controls the mechanics of my iron beads sorting machine
// the machine has two stepper motors
// "stepper" is the motor to turn the "beads isolator", which moves bead per bead to a camera
// "stepperDrehteller" is the motor to move the turntable with the bins to collect the beads
// each bin shall receive beads of one color
// the camera takes a picture of the beads in the isolator
// the PC reads the camera image, determines the color and sends a bin number via serial to the 
// arduino. the arduino then moves the turntable to the correct position / bin number and moves the 
// isolator such that the bead is droppen into the bin on the turntable. 

#include <AccelStepper.h>

// Define the steppers and the pins they will use
AccelStepper stepper(AccelStepper::DRIVER, 9, 8); // (9=step; 8 = dir)


// values for turntable
int no_bins = 14;
 int bin_pos[14] = { 0, 114,229,343,457,571,686,800,914,1029,1143,1257,1371,1486 }; // 1600 steps per revolution

////////////////////////////////////////////////////////
///////  SETUP
////////////////////////////////////////////////////////
void setup() {
  stepper.setMaxSpeed(4000);
  stepper.setAcceleration(20000);
  stepper.setCurrentPosition(0);

  // stepperDrehteller.setMaxSpeed(4000);
  // stepperDrehteller.setAcceleration(2000);
  // stepperDrehteller.setCurrentPosition(0);

  Serial.begin(115200);

  while (!Serial);  // wait for serial port to connect. Needed for native USB port only

  Serial.println("\nWaiting for serial start signal");
  while (Serial.available() == 0); 
  while (Serial.available() != 0) Serial.read(); 

  Serial.print("Setup finished\n");
}

//////////////////////////////////////////////////////////////////////////
// log text to serial interface
void log2serial(String iv_SerLogText){
  if (0==1 && Serial){
    Serial.println(String(">>>Log: ") + iv_SerLogText);
  }
}

//////////////////////////////////////////////////////////////////////////
// wait4command notifies the serial commmunication partner and 
// waits for a command via serial interface 

String wait4command(void){
  char   lv_inChar;
  String lv_string = "";
  String lv_collectString = ""; 

// Read command from serial input:
  Serial.println("waiting for command");
  while (lv_string == ""){
    while (Serial.available() == 0); // wait for serial input
    // stepperDrehteller.run(); // this second call to run() seems to hinder most of the unwanted movements?!;  
    lv_inChar = Serial.read();
    if (isDigit (lv_inChar) || isAlpha(lv_inChar) || lv_inChar == '\n'){
      // convert the incoming byte to a char and add it to the string:
      lv_collectString.concat(String(lv_inChar));

      // if you get a newline, print the string, then the string's value:
      if (lv_inChar == '\n') {
        lv_string = lv_collectString;
        log2serial ("instring:" + lv_string);
        lv_collectString	= ""; 
        lv_inChar = '\0';
      }
    }
  }

  return lv_string;
}


///////////////////////////////////////////////////////////////////
long stepsPerRev = 1600; // 200 steps per stepper motor revolution

void SetmoveRevolutions(float iv_revs, AccelStepper *stepper) {
	long stepsToGo = static_cast<long>((static_cast<float>(stepsPerRev))*iv_revs);
	stepper->move(stepsToGo);
}

////////////////////////////////////////////////////////////////
// move the bead isolator
void move_isolator(float iv_revs){
  log2serial ("Vereinzelung_Move"); 
  stepper.setMaxSpeed(40000);
  SetmoveRevolutions (iv_revs, &stepper);   
  stepper.setSpeed(400);  // not too fast to avoid slack in the stepper motor
  while (stepper.distanceToGo()>0 || stepper.distanceToGo()<0)   {
    stepper.runSpeedToPosition();
    // stepperDrehteller.run();
  }
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// MAIN LOOP
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

int last_bin = 0;
int i = 0;
int bins_to_go = 0;  
int pos_to_move = 0;
int last_pos = 0;

void loop() {


  log2serial ("Loop_Start");   
  String inString = "";  // string to hold input

  // clear the string for new input:
  inString = wait4command();

  //--------------------------------------------------------------
  // position the turntable
  if (inString.endsWith("\n") && inString.startsWith("b")){  // bin selected
    Serial.print("  last_bin: "); Serial.println(last_bin);
    i = inString.substring(1).toInt();
    Serial.print("....bin to reach: "); Serial.print(i); Serial.print("[");Serial.print(bin_pos[i]); Serial.print("]");
    Serial.print("  last_bin: "); Serial.print(last_bin);
    
    if (i == last_bin){  // nothing to do !
      delay (300);
    } else  {
      // calculate shortest way to next position
      if ( (i-last_bin) > (no_bins/2)) {
      // we would go more than half of a revolution -> take the shorter path!
        bins_to_go = i - last_bin - no_bins;
      } else if ((i - last_bin) < (-no_bins/2) )
        bins_to_go = i - last_bin + no_bins;
      else {
        bins_to_go = i-last_bin; 
      }
      

      // correct direction
      if (bins_to_go>0) pos_to_move = bin_pos[bins_to_go]; else pos_to_move = - bin_pos[abs(bins_to_go)];

      Serial.print ("   bins2go:"); Serial.print (bins_to_go);
      Serial.print ("   pos2move:"); Serial.print (pos_to_move);

    AccelStepper stepperDrehteller(AccelStepper::DRIVER, 6,5);   // 6= step; 5 = Dir  (turntable)
    stepperDrehteller.setMaxSpeed(4000);
    stepperDrehteller.setAcceleration(2000);
    stepperDrehteller.setCurrentPosition(0);
    
      stepperDrehteller.stop();
      stepperDrehteller.setCurrentPosition(0);
      while (stepperDrehteller.distanceToGo()>0 || stepperDrehteller.distanceToGo() < 0)
        stepperDrehteller.run(); // this second call to run() seems to hinder most of the unwanted movements?!

      stepperDrehteller.move (pos_to_move);
      stepperDrehteller.setSpeed(500);   // must be called after moveTo for constant speed; do not move too fast

      // somehow most often the stepper makes a short unwanted movement
      while (stepperDrehteller.distanceToGo()>0 || stepperDrehteller.distanceToGo() < 0)
        stepperDrehteller.runToPosition(); // run with accelleration settings to avoid "motor slack"
      while (stepperDrehteller.distanceToGo()>0 || stepperDrehteller.distanceToGo() < 0)
        stepperDrehteller.run(); // this second call to run() seems to hinder most of the unwanted movements?!
      stepperDrehteller.stop();

      Serial.print("....bin reached: "); Serial.print(i); Serial.print("[");Serial.print(bin_pos[i]); Serial.println("]");
      last_pos = bin_pos[i];
      last_bin = i;
      Serial.print(" new last_bin: "); Serial.println(last_bin);
    }

  } else  if (inString.endsWith("\n") && inString.startsWith("m")){  // move to eject bead
    // move the isolator to eject the bead and avoid another color measurement
    // this also adds some time for the turntable to really stop mechanically
    for (int j = 0; j<6; j++){
          move_isolator(0.0125);
          delay(10);   
    }
    move_isolator(0.05);
    delay(200);  
  } else if (inString != ""){  
    //---------------------------------------------------------------------
    // Vereinzelung drehen ; any command from serial rotates the bead "isolator" 
    move_isolator(0.0025);
    delay(20);
    move_isolator(0.0025);
  }
      
}